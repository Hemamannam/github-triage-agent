from __future__ import annotations

from typing import Any

from fakes import make_context, make_issue, make_pr
from test_classification import bug_assessment
from triage_agent.agent.planner import plan_issue_actions, plan_pr_actions
from triage_agent.agent.state import (
    ActionKind,
    ActionRisk,
    Classification,
    DuplicateVerdict,
    PRAssessment,
    TriageState,
)
from triage_agent.config import AppConfig
from triage_agent.github.models import Label, User
from triage_agent.policies.models import DuplicateClosing, RepositoryPolicy


def base_state(**overrides: Any) -> TriageState:
    defaults: dict[str, Any] = {
        "owner": "acme",
        "repo": "backend",
        "issue_number": 42,
        "issue_data": make_issue(42),
        "repository_context": make_context(),
        "issue_assessment": bug_assessment(),
    }
    defaults.update(overrides)
    return TriageState.model_validate(defaults)


def kinds(actions: list[Any]) -> list[ActionKind]:
    return [a.kind for a in actions]


def test_labels_planned_with_policy_priority_label() -> None:
    state = base_state()
    actions = plan_issue_actions(state, RepositoryPolicy(), AppConfig())
    label_action = next(a for a in actions if a.kind is ActionKind.ADD_LABELS)
    # suggested bug+P1; priority high maps to P1 which is already suggested
    assert label_action.params["labels"] == ["bug", "P1"]
    assert label_action.risk is ActionRisk.SAFE
    assert label_action.requires_approval is True  # planner never self-approves


def test_already_applied_labels_not_replanned() -> None:
    state = base_state(
        issue_data=make_issue(42, labels=[Label(name="bug"), Label(name="P1")])
    )
    actions = plan_issue_actions(state, RepositoryPolicy(), AppConfig())
    assert ActionKind.ADD_LABELS not in kinds(actions)


def test_missing_info_comment_planned_for_incomplete_bug() -> None:
    state = base_state(
        issue_assessment=bug_assessment(
            missing_information=["package version", "complete traceback"]
        )
    )
    actions = plan_issue_actions(state, RepositoryPolicy(), AppConfig())
    comment = next(a for a in actions if a.kind is ActionKind.POST_COMMENT)
    assert "package version" in comment.params["comment"]
    assert "complete traceback" in comment.params["comment"]
    assert "missing-info" in comment.params["marker"]


def test_drafted_comment_overrides_template() -> None:
    state = base_state(
        issue_assessment=bug_assessment(missing_information=["version"])
    )
    actions = plan_issue_actions(
        state, RepositoryPolicy(), AppConfig(), drafted_comment="Custom LLM wording."
    )
    comment = next(a for a in actions if a.kind is ActionKind.POST_COMMENT)
    assert comment.params["comment"] == "Custom LLM wording."


def test_duplicate_close_planned_only_above_threshold() -> None:
    high = base_state(
        duplicate_verdict=DuplicateVerdict(
            duplicate_of=7, confidence=0.96, reason="same exception"
        )
    )
    actions = plan_issue_actions(high, RepositoryPolicy(), AppConfig())
    assert ActionKind.CLOSE_ISSUE in kinds(actions)
    close = next(a for a in actions if a.kind is ActionKind.CLOSE_ISSUE)
    assert close.risk is ActionRisk.DESTRUCTIVE
    assert close.evidence  # verdict reason carried as evidence

    low = base_state(
        duplicate_verdict=DuplicateVerdict(
            duplicate_of=7, confidence=0.80, reason="maybe related"
        )
    )
    actions_low = plan_issue_actions(low, RepositoryPolicy(), AppConfig())
    assert ActionKind.CLOSE_ISSUE not in kinds(actions_low)


def test_duplicate_close_respects_policy_disable() -> None:
    state = base_state(
        duplicate_verdict=DuplicateVerdict(
            duplicate_of=7, confidence=0.99, reason="same exception"
        )
    )
    policy = RepositoryPolicy(duplicate_closing=DuplicateClosing.DISABLED)
    actions = plan_issue_actions(state, policy, AppConfig())
    assert ActionKind.CLOSE_ISSUE not in kinds(actions)


def test_assignment_from_codeowner_users_only() -> None:
    state = base_state(
        codeowner_suggestions=["@backend-team/x", "@alice", "@acme/team", "@bob"]
    )
    actions = plan_issue_actions(state, RepositoryPolicy(), AppConfig())
    assign = next(a for a in actions if a.kind is ActionKind.ASSIGN_ISSUE)
    # Teams cannot be issue assignees; only users are planned, capped at 2.
    assert assign.params["assignees"] == ["alice", "bob"]


def test_no_assignment_when_already_assigned() -> None:
    state = base_state(
        issue_data=make_issue(42, assignees=[User(login="carol")]),
        codeowner_suggestions=["@alice"],
    )
    actions = plan_issue_actions(state, RepositoryPolicy(), AppConfig())
    assert ActionKind.ASSIGN_ISSUE not in kinds(actions)


def test_pr_planning_feedback_and_reviewers() -> None:
    state = base_state(
        object_type="pull_request",
        pr_data=make_pr(42),
        pr_assessment=PRAssessment(
            classification=Classification.FEATURE_REQUEST,
            confidence=0.9,
            summary="Adds limiter",
            reasoning_summary="r",
            missing_tests=True,
            ci_failing=True,
            suggested_labels=["enhancement"],
        ),
        codeowner_suggestions=["@alice", "@carol"],
    )
    actions = plan_pr_actions(state, RepositoryPolicy(), AppConfig())
    assert kinds(actions) == [
        ActionKind.ADD_LABELS,
        ActionKind.POST_COMMENT,
        ActionKind.REQUEST_REVIEW,
    ]
    comment = actions[1].params["comment"]
    assert "no tests" in comment
    assert "CI checks are currently failing" in comment
    # PR author never asked to review their own PR.
    review = actions[2]
    assert "carol" not in review.params["reviewers"]
    assert review.params["reviewers"] == ["alice"]


def test_pr_never_gets_merge_or_approve_actions() -> None:
    """The action vocabulary itself cannot express approve/merge."""
    assert not hasattr(ActionKind, "APPROVE_PR")
    assert not hasattr(ActionKind, "MERGE_PR")
