"""MCP tool implementations.

``TriageToolService`` is transport-agnostic: it validates inputs, enforces
dry-run and confirmation gates, and delegates to the GitHub client. The MCP
server in ``server.py`` is a thin wiring layer over this class, which keeps
every safety property unit-testable without an MCP transport.

Safety model:
    * every input is validated against strict Pydantic schemas — the LLM can
      only reach predefined tools with predefined shapes, never raw API paths
    * ``close_issue`` / ``reopen_issue`` are destructive and require an
      explicit ``confirm=True`` from a human; the first call always returns a
      proposed-action payload and performs no mutation
    * in dry-run mode no mutation ever reaches GitHub; tools report what they
      *would* do instead
"""

from __future__ import annotations

from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

from triage_agent.github.client import GitHubClient
from triage_agent.github.models import RepositoryContext

logger = structlog.get_logger("mcp.tools")

OWNER_PATTERN = r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$"
REPO_PATTERN = r"^[A-Za-z0-9._-]{1,100}$"
USERNAME_PATTERN = OWNER_PATTERN

MAX_COMMENT_LENGTH = 60_000
MAX_TEXT_SNIPPET = 6_000  # cap README/CONTRIBUTING text returned to clients


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepoRef(_StrictModel):
    owner: str = Field(pattern=OWNER_PATTERN)
    repo: str = Field(pattern=REPO_PATTERN)


class IssueRef(RepoRef):
    issue_number: int = Field(ge=1, le=10_000_000)


class SearchInput(RepoRef):
    query: str = Field(min_length=1, max_length=256)


class AddLabelsInput(IssueRef):
    labels: list[str] = Field(min_length=1, max_length=10)

    def model_post_init(self, __context: Any) -> None:
        for label in self.labels:
            if not label.strip() or len(label) > 50:
                raise ValueError(f"invalid label name: {label!r}")


class PostCommentInput(IssueRef):
    comment: str = Field(min_length=1, max_length=MAX_COMMENT_LENGTH)


class AssignInput(IssueRef):
    assignees: list[str] = Field(min_length=1, max_length=10)

    def model_post_init(self, __context: Any) -> None:
        import re

        for name in self.assignees:
            if not re.match(USERNAME_PATTERN, name):
                raise ValueError(f"invalid GitHub username: {name!r}")


class CloseIssueInput(IssueRef):
    reason: str = Field(min_length=1, max_length=1000)
    state_reason: Literal["completed", "not_planned"] = "not_planned"
    confirm: bool = False


class ReopenIssueInput(IssueRef):
    reason: str = Field(min_length=1, max_length=1000)
    confirm: bool = False


def _truncate(text: str | None, limit: int = MAX_TEXT_SNIPPET) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} chars total]"


class TriageToolService:
    """Validated, safety-gated GitHub operations exposed as MCP tools."""

    def __init__(self, client: GitHubClient, *, dry_run: bool = True) -> None:
        self._client = client
        self.dry_run = dry_run

    # ------------------------------------------------------------- read tools

    async def get_issue(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        ref = IssueRef(owner=owner, repo=repo, issue_number=issue_number)
        issue = await self._client.get_issue(ref.owner, ref.repo, ref.issue_number)
        payload = issue.model_dump(mode="json")
        payload["repository"] = f"{ref.owner}/{ref.repo}"
        return payload

    async def search_issues(self, owner: str, repo: str, query: str) -> dict[str, Any]:
        inp = SearchInput(owner=owner, repo=repo, query=query)
        result = await self._client.search_issues(inp.owner, inp.repo, inp.query)
        return result.model_dump(mode="json")

    async def get_repository_context(self, owner: str, repo: str) -> dict[str, Any]:
        ref = RepoRef(owner=owner, repo=repo)
        ctx: RepositoryContext = await self._client.get_repository_context(ref.owner, ref.repo)
        payload = ctx.model_dump(mode="json")
        payload["readme"] = _truncate(ctx.readme)
        payload["contributing"] = _truncate(ctx.contributing)
        return payload

    async def get_pull_request(self, owner: str, repo: str, pr_number: int) -> dict[str, Any]:
        ref = IssueRef(owner=owner, repo=repo, issue_number=pr_number)
        pr = await self._client.get_pull_request(ref.owner, ref.repo, ref.issue_number)
        payload = pr.model_dump(mode="json")
        for change in payload.get("changed_files", []):
            change["patch"] = _truncate(change.get("patch"), 1_500)
        return payload

    async def get_recent_repository_activity(
        self, owner: str, repo: str, days: int = 14
    ) -> dict[str, Any]:
        ref = RepoRef(owner=owner, repo=repo)
        days = max(1, min(days, 90))
        from datetime import UTC, datetime, timedelta

        since = (datetime.now(UTC) - timedelta(days=days)).date().isoformat()
        result = await self._client.search_issues(
            ref.owner, ref.repo, f"updated:>={since} sort:updated-desc"
        )
        return result.model_dump(mode="json")

    # --------------------------------------------------------- mutating tools

    async def add_labels(
        self, owner: str, repo: str, issue_number: int, labels: list[str]
    ) -> dict[str, Any]:
        inp = AddLabelsInput(owner=owner, repo=repo, issue_number=issue_number, labels=labels)
        if self.dry_run:
            return self._would("add_labels", inp, f"add labels {inp.labels} to #{inp.issue_number}")
        applied = await self._client.add_labels(
            inp.owner, inp.repo, inp.issue_number, inp.labels
        )
        return {"status": "ok", "labels": [label.name for label in applied]}

    async def post_comment(
        self, owner: str, repo: str, issue_number: int, comment: str
    ) -> dict[str, Any]:
        inp = PostCommentInput(
            owner=owner, repo=repo, issue_number=issue_number, comment=comment
        )
        if self.dry_run:
            return self._would(
                "post_comment", inp, f"post a {len(inp.comment)}-char comment on #{inp.issue_number}"
            )
        posted = await self._client.post_comment(
            inp.owner, inp.repo, inp.issue_number, inp.comment
        )
        return {"status": "ok", "comment_id": posted.id}

    async def assign_issue(
        self, owner: str, repo: str, issue_number: int, assignees: list[str]
    ) -> dict[str, Any]:
        inp = AssignInput(
            owner=owner, repo=repo, issue_number=issue_number, assignees=assignees
        )
        # Never assign unverifiable users — the model must not hallucinate names.
        unverified = [
            name
            for name in inp.assignees
            if not await self._client.is_assignable(inp.owner, inp.repo, name)
        ]
        if unverified:
            return {
                "status": "rejected",
                "reason": f"users not assignable in {inp.owner}/{inp.repo}: {unverified}",
            }
        if self.dry_run:
            return self._would(
                "assign_issue", inp, f"assign {inp.assignees} to #{inp.issue_number}"
            )
        updated = await self._client.assign_issue(
            inp.owner, inp.repo, inp.issue_number, inp.assignees
        )
        return {"status": "ok", "assignees": [u.login for u in updated.assignees]}

    # ------------------------------------------------- destructive (gated)

    async def close_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        reason: str,
        state_reason: str = "not_planned",
        confirm: bool = False,
    ) -> dict[str, Any]:
        inp = CloseIssueInput(
            owner=owner,
            repo=repo,
            issue_number=issue_number,
            reason=reason,
            state_reason=state_reason,  # type: ignore[arg-type]
            confirm=confirm,
        )
        proposed = {
            "action": "close_issue",
            "target": f"{inp.owner}/{inp.repo}#{inp.issue_number}",
            "state_reason": inp.state_reason,
            "reason": inp.reason,
        }
        if not inp.confirm:
            logger.info("close_issue.confirmation_required", **proposed)
            return {
                "status": "confirmation_required",
                "proposed_action": proposed,
                "instructions": (
                    "Closing an issue is destructive and requires explicit human "
                    "approval. Show the proposed action to the user; only if they "
                    "approve, call close_issue again with confirm=true."
                ),
            }
        if self.dry_run:
            return self._would(
                "close_issue", inp, f"close #{inp.issue_number} as {inp.state_reason}"
            )
        closed = await self._client.close_issue(
            inp.owner, inp.repo, inp.issue_number, state_reason=inp.state_reason
        )
        logger.info("close_issue.executed", **proposed)
        return {"status": "ok", "state": closed.state.value, "reason": inp.reason}

    async def reopen_issue(
        self, owner: str, repo: str, issue_number: int, reason: str, confirm: bool = False
    ) -> dict[str, Any]:
        inp = ReopenIssueInput(
            owner=owner, repo=repo, issue_number=issue_number, reason=reason, confirm=confirm
        )
        if not inp.confirm:
            return {
                "status": "confirmation_required",
                "proposed_action": {
                    "action": "reopen_issue",
                    "target": f"{inp.owner}/{inp.repo}#{inp.issue_number}",
                    "reason": inp.reason,
                },
                "instructions": (
                    "Reopening an issue requires explicit human approval. "
                    "Call again with confirm=true only after the user approves."
                ),
            }
        if self.dry_run:
            return self._would("reopen_issue", inp, f"reopen #{inp.issue_number}")
        reopened = await self._client.reopen_issue(inp.owner, inp.repo, inp.issue_number)
        return {"status": "ok", "state": reopened.state.value, "reason": inp.reason}

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _would(action: str, inp: RepoRef, description: str) -> dict[str, Any]:
        logger.info("dry_run.would_execute", action=action, description=description)
        return {
            "status": "dry_run",
            "would_execute": f"{description} in {inp.owner}/{inp.repo}",
        }
