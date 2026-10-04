"""Deterministic action planning.

The planner converts validated assessments into ``ProposedAction``s. It is
pure logic — every proposal carries its reason and evidence, labels come only
from the repository's label set, assignees only from CODEOWNERS, and thresholds
come from configuration. The LLM's only optional contribution here is comment
wording (drafted upstream and passed in as text).
"""

from __future__ import annotations

import structlog

from triage_agent.agent.labels import filter_labels
from triage_agent.agent.state import (
    ACTION_BASE_RISK,
    ActionKind,
    Classification,
    ProposedAction,
    TriageState,
)
from triage_agent.config import AppConfig
from triage_agent.github.codeowners import split_owner
from triage_agent.policies.engine import PolicyEngine
from triage_agent.policies.models import RepositoryPolicy

logger = structlog.get_logger("agent.planner")

# Hidden HTML markers make comments idempotent: the executor never posts a
# second comment carrying the same marker on the same issue.
MARKER_PREFIX = "github-triage-agent"


def comment_marker(tag: str) -> str:
    return f"<!-- {MARKER_PREFIX}:{tag} -->"


def missing_info_comment(items: list[str]) -> str:
    bullets = "\n".join(f"- {item}" for item in items)
    return (
        "Thanks for reporting this! To help us investigate, could you add:\n"
        f"{bullets}\n\n"
        "_— automated triage; a maintainer will follow up once this is added_"
    )


def pr_feedback_comment(
    *, missing_tests: bool, missing_docs: bool, ci_failing: bool, unusually_large: bool
) -> str:
    points: list[str] = []
    if missing_tests:
        points.append("- This change modifies code but no tests were added or updated.")
    if missing_docs:
        points.append("- User-facing behaviour changed but documentation was not updated.")
    if ci_failing:
        points.append("- CI checks are currently failing on this branch.")
    if unusually_large:
        points.append(
            "- This PR is unusually large; consider splitting it to make review easier."
        )
    return (
        "Thanks for the contribution! A few automated observations:\n"
        + "\n".join(points)
        + "\n\n_— automated triage_"
    )


def _make_action(
    kind: ActionKind, params: dict[str, object], reason: str, evidence: list[str] | None = None
) -> ProposedAction:
    return ProposedAction(
        kind=kind,
        params=params,
        reason=reason,
        risk=ACTION_BASE_RISK[kind],
        requires_approval=True,  # the safety layer explicitly relaxes this
        evidence=evidence or [],
    )


def _codeowner_users(suggestions: list[str], limit: int = 2) -> list[str]:
    users = [name for owner in suggestions if (kind_name := split_owner(owner))[0] == "user"
             for name in [kind_name[1]]]
    return users[:limit]


def plan_issue_actions(
    state: TriageState,
    policy: RepositoryPolicy,
    config: AppConfig,
    *,
    drafted_comment: str | None = None,
) -> list[ProposedAction]:
    assessment = state.issue_assessment
    issue = state.issue_data
    if assessment is None or issue is None:
        return []
    actions: list[ProposedAction] = []

    # 1. Labels (suggested + policy priority label), only ones not yet applied.
    #    Re-filtered against the repo's real labels — defence in depth.
    labels = filter_labels(assessment.suggested_labels, state.repository_context)
    priority_label = PolicyEngine.priority_label(policy, assessment.priority)
    if priority_label and state.repository_context and (
        priority_label in state.repository_context.available_label_names
    ) and priority_label not in labels:
        labels.append(priority_label)
    new_labels = [name for name in labels if name not in issue.label_names]
    if new_labels:
        actions.append(
            _make_action(
                ActionKind.ADD_LABELS,
                {"labels": new_labels},
                f"Classified as {assessment.classification.value} "
                f"(priority {assessment.priority.value}).",
            )
        )

    # 2. Request missing information on incomplete bug reports.
    if (
        assessment.classification is Classification.BUG
        and assessment.missing_information
    ):
        body = drafted_comment or missing_info_comment(assessment.missing_information)
        actions.append(
            _make_action(
                ActionKind.POST_COMMENT,
                {"comment": body, "marker": comment_marker("missing-info")},
                "Bug report is missing information needed to debug.",
                evidence=[f"missing: {item}" for item in assessment.missing_information],
            )
        )

    # 3. Duplicate handling — only above the configured close threshold.
    verdict = state.duplicate_verdict
    if (
        verdict is not None
        and verdict.duplicate_of is not None
        and verdict.confidence >= config.duplicates.close_threshold
        and PolicyEngine.duplicate_closing_enabled(policy)
    ):
        dup = verdict.duplicate_of
        evidence = [verdict.reason] + [
            f"lexical similarity {c.similarity:.2f}"
            for c in state.duplicate_candidates
            if c.issue_number == dup
        ]
        actions.append(
            _make_action(
                ActionKind.POST_COMMENT,
                {
                    "comment": (
                        f"This looks like a duplicate of #{dup} "
                        f"(confidence {verdict.confidence:.0%}). {verdict.reason}\n\n"
                        "_— automated triage_"
                    ),
                    "marker": comment_marker("duplicate"),
                },
                f"Explains the duplicate verdict before closing as duplicate of #{dup}.",
                evidence=evidence,
            )
        )
        actions.append(
            _make_action(
                ActionKind.CLOSE_ISSUE,
                {"state_reason": "not_planned"},
                f"Likely duplicate of #{dup} with {verdict.confidence:.0%} confidence.",
                evidence=evidence,
            )
        )

    # 4. Close clearly-invalid issues (still destructive => approval-gated).
    elif (
        assessment.classification is Classification.INVALID
        and assessment.confidence >= config.automation.safe_action_threshold
    ):
        actions.append(
            _make_action(
                ActionKind.CLOSE_ISSUE,
                {"state_reason": "not_planned"},
                f"Classified invalid with {assessment.confidence:.0%} confidence: "
                f"{assessment.reasoning_summary[:200]}",
            )
        )

    # 5. CODEOWNERS-grounded assignment (users only; verified at execution).
    assignees = [
        user
        for user in _codeowner_users(state.codeowner_suggestions)
        if user not in [u.login for u in issue.assignees]
    ]
    if assignees and not issue.assignees:
        actions.append(
            _make_action(
                ActionKind.ASSIGN_ISSUE,
                {"assignees": assignees},
                "CODEOWNERS maps this component to these maintainers.",
                evidence=[f"codeowners suggestions: {state.codeowner_suggestions}"],
            )
        )

    logger.info("planned.issue_actions", count=len(actions))
    return actions


def plan_pr_actions(
    state: TriageState,
    policy: RepositoryPolicy,
    config: AppConfig,
) -> list[ProposedAction]:
    assessment = state.pr_assessment
    pr = state.pr_data
    if assessment is None or pr is None:
        return []
    actions: list[ProposedAction] = []

    new_labels = [
        name
        for name in filter_labels(assessment.suggested_labels, state.repository_context)
        if name not in pr.label_names
    ]
    if new_labels:
        actions.append(
            _make_action(
                ActionKind.ADD_LABELS,
                {"labels": new_labels},
                f"PR classified as {assessment.classification.value}.",
            )
        )

    if (
        assessment.missing_tests
        or assessment.missing_documentation
        or assessment.ci_failing
        or assessment.unusually_large
    ):
        actions.append(
            _make_action(
                ActionKind.POST_COMMENT,
                {
                    "comment": pr_feedback_comment(
                        missing_tests=assessment.missing_tests,
                        missing_docs=assessment.missing_documentation,
                        ci_failing=assessment.ci_failing,
                        unusually_large=assessment.unusually_large,
                    ),
                    "marker": comment_marker("pr-feedback"),
                },
                "PR is missing tests/docs or has failing CI.",
            )
        )

    reviewers = [
        user
        for user in _codeowner_users(state.codeowner_suggestions)
        if user != pr.user.login  # never ask the author to review themselves
    ]
    if reviewers and not pr.reviews:
        actions.append(
            _make_action(
                ActionKind.REQUEST_REVIEW,
                {"reviewers": reviewers},
                "CODEOWNERS maps the changed files to these maintainers.",
                evidence=[f"codeowners suggestions: {state.codeowner_suggestions}"],
            )
        )

    logger.info("planned.pr_actions", count=len(actions))
    return actions
