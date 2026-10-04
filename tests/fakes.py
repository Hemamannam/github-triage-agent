"""Test doubles shared across phases.

``FakeGitHubClient`` duck-types the surface of ``GitHubClient`` used by the
tool service and the agent graph, records every mutation, and serves canned
data — so tests can assert on exactly which side effects happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from triage_agent.github.client import RetryMetrics
from triage_agent.github.models import (
    Comment,
    IssueData,
    Label,
    PullRequestData,
    RepoInfo,
    RepositoryContext,
    SearchResult,
    User,
)


def make_issue(number: int = 42, **overrides: Any) -> IssueData:
    defaults: dict[str, Any] = {
        "number": number,
        "title": "OAuth callback crashes when token expires",
        "body": "Traceback: KeyError in refresh_token. Version 2.1.0, Python 3.12.",
        "state": "open",
        "user": User(login="alice"),
        "labels": [],
        "html_url": f"https://github.com/acme/backend/issues/{number}",
    }
    defaults.update(overrides)
    return IssueData.model_validate(defaults)


def make_pr(number: int = 10, **overrides: Any) -> PullRequestData:
    defaults: dict[str, Any] = {
        "number": number,
        "title": "Add rate limiter",
        "body": "Implements request throttling. Fixes #42",
        "state": "open",
        "user": User(login="carol"),
        "additions": 120,
        "deletions": 5,
        "html_url": f"https://github.com/acme/backend/pull/{number}",
    }
    defaults.update(overrides)
    return PullRequestData.model_validate(defaults)


def make_context(**overrides: Any) -> RepositoryContext:
    defaults: dict[str, Any] = {
        "info": RepoInfo(full_name="acme/backend", description="Backend service"),
        "readme": "# Backend",
        "contributing": "Please include reproduction steps.",
        "codeowners": "/backend/ @backend-team\n/docs/ @docs-team",
        "available_labels": [
            Label(name="bug"),
            Label(name="enhancement"),
            Label(name="documentation"),
            Label(name="question"),
            Label(name="P0"),
            Label(name="P1"),
            Label(name="P2"),
            Label(name="P3"),
        ],
    }
    defaults.update(overrides)
    return RepositoryContext.model_validate(defaults)


@dataclass
class FakeGitHubClient:
    issues: dict[int, IssueData] = field(default_factory=dict)
    prs: dict[int, PullRequestData] = field(default_factory=dict)
    context: RepositoryContext = field(default_factory=make_context)
    search_results: SearchResult = field(default_factory=SearchResult)
    assignable_users: set[str] = field(default_factory=lambda: {"alice", "bob"})
    mutations_enabled: bool = True
    metrics: RetryMetrics = field(default_factory=RetryMetrics)
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    def mutation_calls(self) -> list[str]:
        mutating = {
            "add_labels", "post_comment", "assign_issue",
            "close_issue", "reopen_issue", "request_review",
        }
        return [name for name, _ in self.calls if name in mutating]

    # reads -----------------------------------------------------------------

    async def get_issue(
        self, owner: str, repo: str, issue_number: int, *, with_comments: bool = True
    ) -> IssueData:
        self.calls.append(("get_issue", (owner, repo, issue_number)))
        return self.issues.get(issue_number) or make_issue(issue_number)

    async def search_issues(self, owner: str, repo: str, query: str) -> SearchResult:
        self.calls.append(("search_issues", (owner, repo, query)))
        return self.search_results

    async def get_repository_context(self, owner: str, repo: str) -> RepositoryContext:
        self.calls.append(("get_repository_context", (owner, repo)))
        return self.context

    async def get_pull_request(self, owner: str, repo: str, pr_number: int) -> PullRequestData:
        self.calls.append(("get_pull_request", (owner, repo, pr_number)))
        return self.prs.get(pr_number) or make_pr(pr_number)

    async def is_assignable(self, owner: str, repo: str, username: str) -> bool:
        self.calls.append(("is_assignable", (owner, repo, username)))
        return username in self.assignable_users

    # mutations -------------------------------------------------------------

    async def add_labels(
        self, owner: str, repo: str, issue_number: int, labels: list[str]
    ) -> list[Label]:
        self.calls.append(("add_labels", (owner, repo, issue_number, tuple(labels))))
        return [Label(name=name) for name in labels]

    async def post_comment(
        self, owner: str, repo: str, issue_number: int, body: str
    ) -> Comment:
        self.calls.append(("post_comment", (owner, repo, issue_number, body)))
        return Comment(id=1001, user=User(login="triage-bot"), body=body)

    async def assign_issue(
        self, owner: str, repo: str, issue_number: int, assignees: list[str]
    ) -> IssueData:
        self.calls.append(("assign_issue", (owner, repo, issue_number, tuple(assignees))))
        return make_issue(issue_number, assignees=[User(login=a) for a in assignees])

    async def close_issue(
        self, owner: str, repo: str, issue_number: int, *, state_reason: str = "completed"
    ) -> IssueData:
        self.calls.append(("close_issue", (owner, repo, issue_number, state_reason)))
        return make_issue(issue_number, state="closed")

    async def reopen_issue(self, owner: str, repo: str, issue_number: int) -> IssueData:
        self.calls.append(("reopen_issue", (owner, repo, issue_number)))
        return make_issue(issue_number, state="open")

    async def request_review(
        self, owner: str, repo: str, pr_number: int, reviewers: list[str]
    ) -> None:
        self.calls.append(("request_review", (owner, repo, pr_number, tuple(reviewers))))

    async def aclose(self) -> None:
        return None


class FakeLLM:
    """Scripted LLM backend: queue responses (or exceptions) per schema name."""

    def __init__(self) -> None:
        self._queues: dict[str, list[Any]] = {}
        self.calls: list[tuple[str, str, str]] = []  # (schema, system, prompt)

    def queue(self, response: Any) -> None:
        """Queue a response instance, or an Exception to be raised."""
        name = (
            type(response).__name__
            if not isinstance(response, type) and not isinstance(response, Exception)
            else getattr(response, "schema_name", type(response).__name__)
        )
        self._queues.setdefault(name, []).append(response)

    def queue_error(self, schema_name: str, error: Exception) -> None:
        self._queues.setdefault(schema_name, []).append(error)

    async def structured(self, *, system: str, prompt: str, schema: type) -> Any:
        self.calls.append((schema.__name__, system, prompt))
        queue = self._queues.get(schema.__name__, [])
        if not queue:
            raise AssertionError(
                f"FakeLLM has no queued response for schema {schema.__name__}"
            )
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
