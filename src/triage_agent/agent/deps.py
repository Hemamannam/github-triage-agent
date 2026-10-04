"""Dependency container for the agent graph.

Nodes receive everything they need through ``TriageDeps`` — no globals, no
hidden imports. Protocols keep the graph testable with fakes and make the
GitHub/LLM boundaries explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar

from pydantic import BaseModel

from triage_agent.config import AppConfig
from triage_agent.github.models import (
    IssueData,
    PullRequestData,
    RepositoryContext,
    SearchResult,
)

if TYPE_CHECKING:
    from triage_agent.github.client import RetryMetrics

T = TypeVar("T", bound=BaseModel)


class GitHubAPI(Protocol):
    """The slice of the GitHub client the agent graph depends on."""

    metrics: RetryMetrics

    async def get_issue(
        self, owner: str, repo: str, issue_number: int, *, with_comments: bool = ...
    ) -> IssueData: ...

    async def get_pull_request(
        self, owner: str, repo: str, pr_number: int
    ) -> PullRequestData: ...

    async def get_repository_context(self, owner: str, repo: str) -> RepositoryContext: ...

    async def search_issues(self, owner: str, repo: str, query: str) -> SearchResult: ...

    async def is_assignable(self, owner: str, repo: str, username: str) -> bool: ...

    async def add_labels(
        self, owner: str, repo: str, issue_number: int, labels: list[str]
    ) -> object: ...

    async def post_comment(
        self, owner: str, repo: str, issue_number: int, body: str
    ) -> object: ...

    async def assign_issue(
        self, owner: str, repo: str, issue_number: int, assignees: list[str]
    ) -> object: ...

    async def close_issue(
        self, owner: str, repo: str, issue_number: int, *, state_reason: str = ...
    ) -> object: ...

    async def reopen_issue(self, owner: str, repo: str, issue_number: int) -> object: ...

    async def request_review(
        self, owner: str, repo: str, pr_number: int, reviewers: list[str]
    ) -> object: ...


class LLMBackend(Protocol):
    """A model backend that returns schema-validated structured output."""

    async def structured(self, *, system: str, prompt: str, schema: type[T]) -> T: ...


@dataclass
class TriageDeps:
    github: GitHubAPI
    config: AppConfig = field(default_factory=AppConfig)
    llm: LLMBackend | None = None
    dry_run: bool = True
    # "interrupt": pause the graph and wait for a human decision (needs a
    # checkpointer). "none": held actions are simply not executed (headless).
    approval_mode: Literal["interrupt", "none"] = "none"
