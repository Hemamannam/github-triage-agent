"""LangGraph node implementations.

Each node is a small async function ``TriageState -> partial state update``.
LLM-dependent nodes (classification, duplicate comparison, action planning)
are introduced in later phases; this module starts with the deterministic
skeleton so the graph is runnable end-to-end from Phase 3 onward.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from langgraph.types import interrupt

from triage_agent.agent import prompts
from triage_agent.agent.deps import TriageDeps
from triage_agent.agent.duplicates import extract_keywords, score_candidates
from triage_agent.agent.executor import ActionExecutor
from triage_agent.agent.labels import filter_labels
from triage_agent.agent.planner import plan_issue_actions, plan_pr_actions
from triage_agent.agent.state import (
    Classification,
    CommentDraft,
    DuplicateVerdict,
    ExecutedAction,
    ExecutionStatus,
    IssueAssessment,
    ObjectType,
    PRAssessment,
    Priority,
    Severity,
    TriageState,
)
from triage_agent.github.codeowners import CodeOwners
from triage_agent.github.models import PullRequestData
from triage_agent.safety.approvals import (
    apply_decisions,
    parse_resume_payload,
    render_approval_request,
)
from triage_agent.safety.guardrails import SafetyGuardrails

logger = structlog.get_logger("agent.nodes")

_TEST_FILE = re.compile(r"(^|/)(tests?|spec)s?/|(^|/)test_|_test\.|\.(spec|test)\.")
_DOC_FILE = re.compile(r"\.(md|rst|adoc)$|(^|/)docs?/", re.IGNORECASE)
_LARGE_PR_LINES = 800
_LARGE_PR_FILES = 20


def pr_signals(pr: PullRequestData) -> dict[str, bool]:
    """Deterministic PR facts computed from the diff and CI — never left to
    the model's imagination."""
    code_files = [
        f.filename
        for f in pr.changed_files
        if not _TEST_FILE.search(f.filename) and not _DOC_FILE.search(f.filename)
    ]
    has_tests = any(_TEST_FILE.search(f.filename) for f in pr.changed_files)
    has_docs = any(_DOC_FILE.search(f.filename) for f in pr.changed_files)
    return {
        "ci_failing": pr.ci_failing,
        "unusually_large": (
            pr.total_diff_size > _LARGE_PR_LINES
            or len(pr.changed_files) > _LARGE_PR_FILES
        ),
        "touches_code": bool(code_files),
        "has_test_changes": has_tests,
        "has_doc_changes": has_docs,
        "missing_tests": bool(code_files) and not has_tests,
    }




class TriageNodes:
    """Node functions bound to their dependencies."""

    def __init__(self, deps: TriageDeps) -> None:
        self.deps = deps

    # ------------------------------------------------------------ fetch nodes

    async def fetch_repository_context(self, state: TriageState) -> dict[str, Any]:
        logger.info("node.fetch_repository_context", repository=state.repository)
        try:
            ctx = await self.deps.github.get_repository_context(state.owner, state.repo)
        except Exception as exc:
            return {"error": f"failed to fetch repository context: {exc}"}
        return {"repository_context": ctx}

    async def fetch_item(self, state: TriageState) -> dict[str, Any]:
        logger.info(
            "node.fetch_item", repository=state.repository, number=state.issue_number
        )
        try:
            issue = await self.deps.github.get_issue(
                state.owner, state.repo, state.issue_number
            )
            if issue.is_pull_request or state.object_type is ObjectType.PULL_REQUEST:
                pr = await self.deps.github.get_pull_request(
                    state.owner, state.repo, state.issue_number
                )
                return {
                    "object_type": ObjectType.PULL_REQUEST,
                    "issue_data": issue,
                    "pr_data": pr,
                }
            return {"object_type": ObjectType.ISSUE, "issue_data": issue}
        except Exception as exc:
            return {"error": f"failed to fetch #{state.issue_number}: {exc}"}

    # --------------------------------------------------------- analysis nodes

    async def classify_request(self, state: TriageState) -> dict[str, Any]:
        """Classify the item via a schema-validated LLM call, with graceful
        degradation: any failure escalates to a human instead of crashing."""
        logger.info("node.classify_request", repository=state.repository)
        if self.deps.llm is None:
            return {
                "issue_assessment": IssueAssessment(
                    classification=Classification.OTHER,
                    confidence=0.0,
                    severity=Severity.NONE,
                    priority=Priority.LOW,
                    reasoning_summary="No LLM backend configured; escalating to a human.",
                ),
                "escalated_to_human": True,
                "escalation_reason": "no LLM backend configured",
            }
        try:
            if state.object_type is ObjectType.PULL_REQUEST and state.pr_data:
                signals = pr_signals(state.pr_data)
                assessment_pr: PRAssessment = await self.deps.llm.structured(
                    system=prompts.PR_CLASSIFY_SYSTEM,
                    prompt=prompts.build_pr_classify_prompt(state, signals),
                    schema=PRAssessment,
                )
                # Objective facts always win over the model's echo of them.
                assessment_pr = assessment_pr.model_copy(
                    update={
                        "ci_failing": signals["ci_failing"],
                        "unusually_large": signals["unusually_large"],
                        "missing_tests": assessment_pr.missing_tests or signals["missing_tests"],
                        "suggested_labels": filter_labels(
                            assessment_pr.suggested_labels, state.repository_context
                        ),
                    }
                )
                logger.info(
                    "classified",
                    classification=assessment_pr.classification.value,
                    confidence=assessment_pr.confidence,
                )
                return {"pr_assessment": assessment_pr}

            assessment: IssueAssessment = await self.deps.llm.structured(
                system=prompts.ISSUE_CLASSIFY_SYSTEM,
                prompt=prompts.build_issue_classify_prompt(state),
                schema=IssueAssessment,
            )
            assessment = assessment.model_copy(
                update={
                    "suggested_labels": filter_labels(
                        assessment.suggested_labels, state.repository_context
                    )
                }
            )
            logger.info(
                "classified",
                classification=assessment.classification.value,
                confidence=assessment.confidence,
            )
            return {"issue_assessment": assessment}
        except Exception as exc:
            logger.error("classify.failed", error=str(exc))
            return {
                "tool_errors": [*state.tool_errors, f"classification failed: {exc}"],
                "escalated_to_human": True,
                "escalation_reason": f"classification failed: {exc}",
            }

    async def gather_additional_context(self, state: TriageState) -> dict[str, Any]:
        """Pull recently-updated issues so later steps can see related work."""
        try:
            from datetime import UTC, datetime, timedelta

            since = (datetime.now(UTC) - timedelta(days=30)).date().isoformat()
            result = await self.deps.github.search_issues(
                state.owner, state.repo, f"updated:>={since} sort:updated-desc"
            )
            recent = [
                item for item in result.items if item.number != state.issue_number
            ][:10]
            return {"recent_activity": recent}
        except Exception as exc:
            # Non-fatal: triage continues without recent-activity context.
            logger.warning("gather_additional_context.failed", error=str(exc))
            return {"tool_errors": [*state.tool_errors, f"recent activity fetch failed: {exc}"]}

    async def search_duplicates(self, state: TriageState) -> dict[str, Any]:
        """Hybrid duplicate detection: keyword search -> lexical ranking ->
        LLM comparison of the strongest candidates -> validated verdict."""
        if state.object_type is ObjectType.PULL_REQUEST or state.issue_data is None:
            return {}
        assessment = state.issue_assessment
        keywords = (assessment.search_keywords if assessment else []) or extract_keywords(
            state.issue_data.title
        )
        if not keywords:
            return {}
        try:
            result = await self.deps.github.search_issues(
                state.owner, state.repo, " ".join(keywords[:5]) + " is:issue"
            )
            candidates = score_candidates(
                state.issue_data,
                result.items,
                max_candidates=self.deps.config.duplicates.max_candidates,
            )
            if not candidates:
                return {"duplicate_candidates": []}

            update: dict[str, Any] = {"duplicate_candidates": candidates}
            if self.deps.llm is not None:
                staged = state.model_copy(update={"duplicate_candidates": candidates})
                verdict: DuplicateVerdict = await self.deps.llm.structured(
                    system=prompts.DUPLICATE_SYSTEM,
                    prompt=prompts.build_duplicate_prompt(staged),
                    schema=DuplicateVerdict,
                )
                valid = {c.issue_number for c in candidates}
                if verdict.duplicate_of is not None and verdict.duplicate_of not in valid:
                    logger.warning(
                        "duplicate.hallucinated_reference", claimed=verdict.duplicate_of
                    )
                    verdict = DuplicateVerdict(
                        duplicate_of=None,
                        confidence=0.0,
                        reason=(
                            f"model referenced #{verdict.duplicate_of}, which was not "
                            "among the retrieved candidates; verdict discarded"
                        ),
                    )
                update["duplicate_candidates"] = [
                    c.model_copy(
                        update={
                            "is_duplicate": c.issue_number == verdict.duplicate_of,
                            "llm_confidence": (
                                verdict.confidence
                                if c.issue_number == verdict.duplicate_of
                                else c.llm_confidence
                            ),
                        }
                    )
                    for c in candidates
                ]
                update["duplicate_verdict"] = verdict
                logger.info(
                    "duplicates.assessed",
                    candidates=len(candidates),
                    duplicate_of=verdict.duplicate_of,
                    confidence=verdict.confidence,
                )
            return update
        except Exception as exc:
            logger.warning("search_duplicates.failed", error=str(exc))
            return {"tool_errors": [*state.tool_errors, f"duplicate search failed: {exc}"]}

    async def assess_priority_and_severity(self, state: TriageState) -> dict[str, Any]:
        """Deterministic post-processing: security escalation and grounded
        CODEOWNERS suggestions. Policy label mapping happens at planning."""
        update: dict[str, Any] = {}
        codeowners = CodeOwners.parse(
            state.repository_context.codeowners if state.repository_context else None
        )

        if state.object_type is ObjectType.PULL_REQUEST and state.pr_data:
            update["codeowner_suggestions"] = codeowners.owners_for_files(
                [f.filename for f in state.pr_data.changed_files]
            )
            return update

        assessment = state.issue_assessment
        if assessment is None:
            return {}
        security = (
            assessment.is_security_sensitive
            or assessment.classification is Classification.SECURITY
        )
        if security:
            severity_order = [Severity.NONE, Severity.LOW, Severity.MEDIUM,
                              Severity.HIGH, Severity.CRITICAL]
            bumped = max(assessment.severity, Severity.HIGH, key=severity_order.index)
            update["issue_assessment"] = assessment.model_copy(
                update={
                    "priority": Priority.CRITICAL,
                    "severity": bumped,
                    "is_security_sensitive": True,
                }
            )
            assessment = update["issue_assessment"]
            logger.info("security.escalated", severity=bumped.value)

        terms = [*assessment.suggested_labels, *assessment.search_keywords]
        update["codeowner_suggestions"] = codeowners.owners_for_terms(terms)
        return update

    # ---------------------------------------------------------- action nodes

    async def determine_actions(self, state: TriageState) -> dict[str, Any]:
        """Plan actions deterministically from the validated assessments.
        The LLM may polish comment wording; it never decides what runs."""
        policy = self.deps.config.policy_for(state.owner, state.repo)

        if state.object_type is ObjectType.PULL_REQUEST:
            actions = plan_pr_actions(state, policy, self.deps.config)
            return {"proposed_actions": actions} if actions else {}

        drafted: str | None = None
        assessment = state.issue_assessment
        if (
            self.deps.llm is not None
            and assessment is not None
            and assessment.classification is Classification.BUG
            and assessment.missing_information
        ):
            try:
                draft: CommentDraft = await self.deps.llm.structured(
                    system=prompts.COMMENT_SYSTEM,
                    prompt=prompts.build_missing_info_prompt(state),
                    schema=CommentDraft,
                )
                drafted = draft.body
            except Exception as exc:
                # Fall back to the deterministic template.
                logger.warning("comment_draft.failed", error=str(exc))

        actions = plan_issue_actions(
            state, policy, self.deps.config, drafted_comment=drafted
        )
        return {"proposed_actions": actions} if actions else {}

    async def safety_check(self, state: TriageState) -> dict[str, Any]:
        """Split proposed actions into auto-approved and held-for-human via
        the guardrails (policy + confidence + risk)."""
        if not state.proposed_actions:
            if state.confidence and state.confidence < self.deps.config.automation.review_threshold:
                return {
                    "escalated_to_human": True,
                    "escalation_reason": (
                        f"low confidence ({state.confidence:.2f}); no automatic action taken"
                    ),
                }
            return {}
        guardrails = SafetyGuardrails(self.deps.config)
        policy = self.deps.config.policy_for(state.owner, state.repo)
        result = guardrails.evaluate(state, policy)
        update: dict[str, Any] = {
            "approved_actions": result.auto_approved,
            "pending_approval": result.pending_approval,
        }
        if result.pending_approval:
            update["escalated_to_human"] = True
            update["escalation_reason"] = "; ".join(
                dict.fromkeys(result.escalation_reasons)
            )
        return update

    async def human_approval(self, state: TriageState) -> dict[str, Any]:
        """Pause for a human decision on held actions.

        interrupt mode: suspends the graph (checkpointer required); the run
        resumes with the human's decisions. Undecided actions are rejected —
        silence never approves anything.
        headless mode: held actions stay unexecuted and are reported.
        """
        if not state.pending_approval:
            return {}
        if self.deps.approval_mode != "interrupt":
            logger.info(
                "approval.headless_hold", held=len(state.pending_approval)
            )
            return {}
        payload = {
            "request": render_approval_request(state, state.pending_approval),
            "actions": [a.model_dump(mode="json") for a in state.pending_approval],
            "run_id": state.run_id,
        }
        resume_value = interrupt(payload)
        decisions = parse_resume_payload(resume_value, state.pending_approval)
        approved, rejected = apply_decisions(state.pending_approval, decisions)
        logger.info(
            "approval.decisions", approved=len(approved), rejected=len(rejected)
        )
        return {
            "approved_actions": [*state.approved_actions, *approved],
            "rejected_actions": [*state.rejected_actions, *rejected],
            "pending_approval": [],
        }

    async def execute_actions(self, state: TriageState) -> dict[str, Any]:
        if not state.approved_actions:
            return {}
        executor = ActionExecutor(self.deps.github, dry_run=self.deps.dry_run)
        executed = await executor.execute(state, state.approved_actions)
        blocked = [
            ExecutedAction(
                action=action,
                status=ExecutionStatus.BLOCKED,
                detail="rejected by safety layer / human review",
            )
            for action in state.rejected_actions
        ]
        failures = [
            f"{e.action.kind.value}: {e.detail}"
            for e in executed
            if e.status is ExecutionStatus.FAILED
        ]
        update: dict[str, Any] = {"executed_actions": [*executed, *blocked]}
        if failures:
            update["tool_errors"] = [*state.tool_errors, *failures]
        return update

    async def verify_actions(self, state: TriageState) -> dict[str, Any]:
        if not state.executed_actions:
            return {}
        executor = ActionExecutor(self.deps.github, dry_run=self.deps.dry_run)
        problems = await executor.verify(state, state.executed_actions)
        if problems:
            logger.warning("verification.problems", problems=problems)
            return {"tool_errors": [*state.tool_errors, *problems]}
        return {}

    # ----------------------------------------------------------- summary node

    async def generate_summary(self, state: TriageState) -> dict[str, Any]:
        lines: list[str] = [
            f"Triage run {state.run_id} — {state.repository}#{state.issue_number}"
        ]
        if state.error:
            lines.append(f"FAILED: {state.error}")
            return {"final_summary": "\n".join(lines)}

        kind = "Pull request" if state.object_type is ObjectType.PULL_REQUEST else "Issue"
        lines.append(f"Type: {kind}")
        if state.classification:
            lines.append(
                f"Classification: {state.classification.value}"
                f" (confidence {state.confidence:.0%})"
            )
        if state.duplicate_verdict and state.duplicate_verdict.duplicate_of:
            lines.append(
                f"Possible duplicate of #{state.duplicate_verdict.duplicate_of}"
                f" (confidence {state.duplicate_verdict.confidence:.0%})"
            )
        for executed in state.executed_actions:
            lines.append(f"[{executed.status.value}] {executed.action.describe()}")
        for held in state.pending_approval:
            lines.append(f"[awaiting human approval] {held.describe()} — {held.reason}")
        if state.escalated_to_human:
            lines.append(f"Escalated to human: {state.escalation_reason}")
        if state.tool_errors:
            lines.append(f"Tool errors: {len(state.tool_errors)}")
        return {"final_summary": "\n".join(lines)}
