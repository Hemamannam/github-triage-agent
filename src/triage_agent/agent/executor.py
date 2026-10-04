"""Action execution with idempotency, error isolation and verification.

Idempotency guards:
    * labels already on the issue are not re-sent (no-op calls avoided)
    * comments carry a hidden HTML marker; if a comment with the same marker
      already exists on the issue, the post is skipped — retries after an
      uncertain network failure cannot double-post
    * assignees are verified via the GitHub assignable check right before the
      call; unverifiable users are dropped, never guessed

Each action executes independently: one failure is recorded and the rest
continue. ``verify`` re-fetches the issue afterwards and checks that the
intended effects actually landed.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

from triage_agent.agent.state import (
    ActionKind,
    ExecutedAction,
    ExecutionStatus,
    ProposedAction,
    TriageState,
)

if TYPE_CHECKING:
    from triage_agent.agent.deps import GitHubAPI

logger = structlog.get_logger("agent.executor")

# Destructive actions run last so a failure earlier in the batch never leaves
# an issue closed-but-unlabelled.
_EXECUTION_ORDER = {
    ActionKind.ADD_LABELS: 0,
    ActionKind.POST_COMMENT: 1,
    ActionKind.ASSIGN_ISSUE: 2,
    ActionKind.REQUEST_REVIEW: 3,
    ActionKind.REOPEN_ISSUE: 4,
    ActionKind.CLOSE_ISSUE: 5,
}


class ActionExecutor:
    def __init__(self, github: GitHubAPI, *, dry_run: bool = True) -> None:
        self._github = github
        self._dry_run = dry_run

    async def execute(
        self, state: TriageState, actions: list[ProposedAction]
    ) -> list[ExecutedAction]:
        results: list[ExecutedAction] = []
        for action in sorted(actions, key=lambda a: _EXECUTION_ORDER[a.kind]):
            start = time.perf_counter()
            try:
                status, detail = await self._execute_one(state, action)
            except Exception as exc:
                status, detail = ExecutionStatus.FAILED, f"{type(exc).__name__}: {exc}"
                logger.error("action.failed", kind=action.kind.value, error=str(exc))
            latency_ms = (time.perf_counter() - start) * 1000
            retries = getattr(self._github.metrics, "last_call_retries", 0)
            results.append(
                ExecutedAction(
                    action=action,
                    status=status,
                    detail=detail,
                    latency_ms=round(latency_ms, 2),
                    retries=retries,
                )
            )
            logger.info(
                "action.executed",
                kind=action.kind.value,
                status=status.value,
                detail=detail[:200],
                latency_ms=round(latency_ms, 2),
            )
        return results

    async def _execute_one(
        self, state: TriageState, action: ProposedAction
    ) -> tuple[ExecutionStatus, str]:
        owner, repo, number = state.owner, state.repo, state.issue_number

        if action.kind is ActionKind.ADD_LABELS:
            current = state.issue_data.label_names if state.issue_data else []
            new_labels = [
                label for label in action.params.get("labels", []) if label not in current
            ]
            if not new_labels:
                return ExecutionStatus.SKIPPED, "all labels already present"
            if self._dry_run:
                return ExecutionStatus.DRY_RUN, f"would add labels {new_labels}"
            await self._github.add_labels(owner, repo, number, new_labels)
            return ExecutionStatus.OK, f"added labels {new_labels}"

        if action.kind is ActionKind.POST_COMMENT:
            marker = str(action.params.get("marker", ""))
            body = str(action.params.get("comment", ""))
            if marker:
                existing = await self._github.get_issue(owner, repo, number)
                if any(marker in c.body for c in existing.comments):
                    return ExecutionStatus.SKIPPED, "comment with this marker already exists"
                body = f"{body}\n\n{marker}"
            if self._dry_run:
                return ExecutionStatus.DRY_RUN, f"would post comment ({len(body)} chars)"
            await self._github.post_comment(owner, repo, number, body)
            return ExecutionStatus.OK, "comment posted"

        if action.kind is ActionKind.ASSIGN_ISSUE:
            requested = list(action.params.get("assignees", []))
            verified = [
                user
                for user in requested
                if await self._github.is_assignable(owner, repo, user)
            ]
            dropped = sorted(set(requested) - set(verified))
            if not verified:
                return (
                    ExecutionStatus.SKIPPED,
                    f"no requested assignee is assignable (dropped {dropped})",
                )
            if self._dry_run:
                return ExecutionStatus.DRY_RUN, f"would assign {verified}"
            await self._github.assign_issue(owner, repo, number, verified)
            detail = f"assigned {verified}"
            if dropped:
                detail += f" (dropped unverifiable {dropped})"
            return ExecutionStatus.OK, detail

        if action.kind is ActionKind.REQUEST_REVIEW:
            reviewers = list(action.params.get("reviewers", []))
            if self._dry_run:
                return ExecutionStatus.DRY_RUN, f"would request review from {reviewers}"
            await self._github.request_review(owner, repo, number, reviewers)
            return ExecutionStatus.OK, f"review requested from {reviewers}"

        if action.kind is ActionKind.CLOSE_ISSUE:
            state_reason = str(action.params.get("state_reason", "not_planned"))
            if self._dry_run:
                return ExecutionStatus.DRY_RUN, f"would close as {state_reason}"
            await self._github.close_issue(owner, repo, number, state_reason=state_reason)
            return ExecutionStatus.OK, f"closed as {state_reason}"

        if action.kind is ActionKind.REOPEN_ISSUE:
            if self._dry_run:
                return ExecutionStatus.DRY_RUN, "would reopen"
            await self._github.reopen_issue(owner, repo, number)
            return ExecutionStatus.OK, "reopened"

        return ExecutionStatus.FAILED, f"unknown action kind {action.kind}"  # pragma: no cover

    async def verify(
        self, state: TriageState, executed: list[ExecutedAction]
    ) -> list[str]:
        """Re-fetch the issue and confirm OK actions actually took effect."""
        if self._dry_run or not any(
            e.status is ExecutionStatus.OK for e in executed
        ):
            return []
        try:
            fresh = await self._github.get_issue(
                state.owner, state.repo, state.issue_number
            )
        except Exception as exc:
            return [f"verification fetch failed: {exc}"]

        problems: list[str] = []
        for item in executed:
            if item.status is not ExecutionStatus.OK:
                continue
            kind, params = item.action.kind, item.action.params
            if kind is ActionKind.ADD_LABELS:
                missing = [
                    label
                    for label in params.get("labels", [])
                    if label not in fresh.label_names
                ]
                if missing:
                    problems.append(f"labels not applied after execution: {missing}")
            elif kind is ActionKind.POST_COMMENT and params.get("marker"):
                if not any(str(params["marker"]) in c.body for c in fresh.comments):
                    problems.append("posted comment not found on issue")
            elif kind is ActionKind.ASSIGN_ISSUE:
                assigned = {u.login for u in fresh.assignees}
                wanted = set(params.get("assignees", []))
                if not (wanted & assigned):
                    problems.append(f"assignees not applied: wanted {sorted(wanted)}")
            elif kind is ActionKind.CLOSE_ISSUE and fresh.state.value != "closed":
                problems.append("issue still open after close")
            elif kind is ActionKind.REOPEN_ISSUE and fresh.state.value != "open":
                problems.append("issue still closed after reopen")
        return problems
