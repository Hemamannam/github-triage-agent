"""Async GitHub REST client with reliability and safety built in.

Reliability:
    * exponential backoff with jitter (tenacity) on transport errors, 5xx and 429
    * GitHub rate-limit awareness (honours ``Retry-After`` / ``X-RateLimit-Reset``)
    * bounded retries, request timeouts, structured error types
    * retry metrics recorded per client for observability

Safety:
    * optional repository allowlist enforced on *every* call
    * mutations can be disabled entirely (``mutations_enabled=False``) —
      dry-run defence-in-depth below the policy layer
    * the token is injected only into request headers and never logged
"""

from __future__ import annotations

import asyncio
import base64
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
)

from triage_agent.config import SecurityConfig
from triage_agent.github.models import (
    CheckRun,
    Comment,
    CommitInfo,
    FileChange,
    IssueData,
    Label,
    PullRequestData,
    RepoInfo,
    RepositoryContext,
    Review,
    SearchResult,
)

logger = structlog.get_logger("github.client")

GITHUB_API = "https://api.github.com"

_CLOSING_KEYWORDS = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)", re.IGNORECASE
)

_CODEOWNERS_PATHS = (".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS")
_CONTRIBUTING_PATHS = ("CONTRIBUTING.md", ".github/CONTRIBUTING.md", "docs/CONTRIBUTING.md")


class GitHubError(Exception):
    """Base class for structured GitHub API errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class GitHubAuthError(GitHubError):
    pass


class GitHubNotFoundError(GitHubError):
    pass


class GitHubRateLimitError(GitHubError):
    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class GitHubServerError(GitHubError):
    pass


class RepositoryNotAllowedError(GitHubError):
    """The repository is outside the configured allowlist."""


class MutationBlockedError(GitHubError):
    """A write was attempted while mutations are disabled (dry run)."""


@dataclass
class RetryMetrics:
    """Aggregate reliability counters, exposed for the audit trail."""

    requests: int = 0
    retries: int = 0
    failures: int = 0
    rate_limit_hits: int = 0
    last_call_retries: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


def _make_wait(base: float, cap: float):  # type: ignore[no-untyped-def]
    """Exponential backoff with jitter; honour server-provided retry delay."""

    def wait(retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, GitHubRateLimitError):
            return min(exc.retry_after, max(cap, 120.0 * base / 2.0))
        import random

        return min(base**retry_state.attempt_number + random.uniform(0, base / 2.0), cap)

    return wait


class GitHubClient:
    """Thin, typed wrapper over the GitHub REST v3 API."""

    def __init__(
        self,
        token: str,
        *,
        security: SecurityConfig | None = None,
        mutations_enabled: bool = True,
        timeout: float = 30.0,
        max_attempts: int = 4,
        backoff_base: float = 2.0,
        backoff_cap: float = 30.0,
        base_url: str = GITHUB_API,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._security = security or SecurityConfig()
        self.mutations_enabled = mutations_enabled
        self._max_attempts = max_attempts
        self._wait = _make_wait(backoff_base, backoff_cap)
        self.metrics = RetryMetrics()
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            transport=transport,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "github-triage-agent",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> GitHubClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ core

    def _check_repo_allowed(self, owner: str, repo: str) -> None:
        if not self._security.repo_allowed(owner, repo):
            raise RepositoryNotAllowedError(
                f"repository {owner}/{repo} is not in the configured allowlist"
            )

    def _check_mutations(self, action: str) -> None:
        if not self.mutations_enabled:
            raise MutationBlockedError(
                f"mutation '{action}' blocked: client is read-only (dry run)"
            )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        code = response.status_code
        if code < 400:
            return
        try:
            detail = response.json().get("message", response.text[:200])
        except Exception:
            detail = response.text[:200]
        if code in (401, 403) and "rate limit" not in detail.lower():
            raise GitHubAuthError(f"GitHub auth/permission error: {detail}", status_code=code)
        if code == 404:
            raise GitHubNotFoundError(f"not found: {detail}", status_code=code)
        if code == 429 or (code == 403 and "rate limit" in detail.lower()):
            retry_after = float(response.headers.get("Retry-After", 0) or 0)
            if not retry_after:
                reset = float(response.headers.get("X-RateLimit-Reset", 0) or 0)
                retry_after = max(reset - time.time(), 1.0) if reset else 15.0
            raise GitHubRateLimitError(f"rate limited: {detail}", retry_after=retry_after)
        if code >= 500:
            raise GitHubServerError(f"GitHub server error {code}: {detail}", status_code=code)
        raise GitHubError(f"GitHub error {code}: {detail}", status_code=code)

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """One API call with bounded, backoff-managed retries."""
        self.metrics.requests += 1
        self.metrics.last_call_retries = 0
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_attempts),
                wait=self._wait,
                retry=retry_if_exception_type(
                    (httpx.TransportError, GitHubServerError, GitHubRateLimitError)
                ),
                reraise=True,
            ):
                with attempt:
                    if attempt.retry_state.attempt_number > 1:
                        self.metrics.retries += 1
                        self.metrics.last_call_retries += 1
                        logger.warning(
                            "github.retry",
                            method=method,
                            url=url,
                            attempt=attempt.retry_state.attempt_number,
                        )
                    response = await self._client.request(method, url, **kwargs)
                    try:
                        self._raise_for_status(response)
                    except GitHubRateLimitError:
                        self.metrics.rate_limit_hits += 1
                        raise
                    return response
        except Exception:
            self.metrics.failures += 1
            raise
        raise GitHubError("retry loop exited unexpectedly")  # pragma: no cover

    async def _get_json(self, url: str, **kwargs: Any) -> Any:
        return (await self._request("GET", url, **kwargs)).json()

    # ----------------------------------------------------------------- reads

    async def get_issue(
        self, owner: str, repo: str, issue_number: int, *, with_comments: bool = True
    ) -> IssueData:
        self._check_repo_allowed(owner, repo)
        raw = await self._get_json(f"/repos/{owner}/{repo}/issues/{issue_number}")
        raw["is_pull_request"] = "pull_request" in raw
        issue = IssueData.model_validate(raw)
        if with_comments and issue.comments_count:
            issue.comments = await self.list_issue_comments(owner, repo, issue_number)
        return issue

    async def list_issue_comments(
        self, owner: str, repo: str, issue_number: int, *, limit: int = 30
    ) -> list[Comment]:
        self._check_repo_allowed(owner, repo)
        raw = await self._get_json(
            f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
            params={"per_page": min(limit, 100)},
        )
        return [Comment.model_validate(c) for c in raw]

    async def search_issues(self, owner: str, repo: str, query: str) -> SearchResult:
        """Search issues/PRs, always scoped to a single repository."""
        self._check_repo_allowed(owner, repo)
        q = f"repo:{owner}/{repo} {query}"
        raw = await self._get_json("/search/issues", params={"q": q, "per_page": 20})
        for item in raw.get("items", []):
            item["is_pull_request"] = "pull_request" in item
        return SearchResult.model_validate(raw)

    async def get_repo(self, owner: str, repo: str) -> RepoInfo:
        self._check_repo_allowed(owner, repo)
        return RepoInfo.model_validate(await self._get_json(f"/repos/{owner}/{repo}"))

    async def list_labels(self, owner: str, repo: str) -> list[Label]:
        self._check_repo_allowed(owner, repo)
        raw = await self._get_json(f"/repos/{owner}/{repo}/labels", params={"per_page": 100})
        return [Label.model_validate(label) for label in raw]

    async def get_file_content(self, owner: str, repo: str, path: str) -> str | None:
        """Return decoded file content, or None if the file does not exist."""
        self._check_repo_allowed(owner, repo)
        try:
            raw = await self._get_json(f"/repos/{owner}/{repo}/contents/{path}")
        except GitHubNotFoundError:
            return None
        if isinstance(raw, list) or raw.get("encoding") != "base64":
            return None
        return base64.b64decode(raw.get("content", "")).decode("utf-8", errors="replace")

    async def get_readme(self, owner: str, repo: str) -> str | None:
        self._check_repo_allowed(owner, repo)
        try:
            raw = await self._get_json(f"/repos/{owner}/{repo}/readme")
        except GitHubNotFoundError:
            return None
        return base64.b64decode(raw.get("content", "")).decode("utf-8", errors="replace")

    async def _first_existing_file(
        self, owner: str, repo: str, paths: tuple[str, ...]
    ) -> str | None:
        for path in paths:
            content = await self.get_file_content(owner, repo, path)
            if content is not None:
                return content
        return None

    async def get_repository_context(self, owner: str, repo: str) -> RepositoryContext:
        """Gather the decision-making context for a repository in one call."""
        self._check_repo_allowed(owner, repo)
        info, readme, contributing, codeowners, labels = await asyncio.gather(
            self.get_repo(owner, repo),
            self.get_readme(owner, repo),
            self._first_existing_file(owner, repo, _CONTRIBUTING_PATHS),
            self._first_existing_file(owner, repo, _CODEOWNERS_PATHS),
            self.list_labels(owner, repo),
        )
        return RepositoryContext(
            info=info,
            readme=readme,
            contributing=contributing,
            codeowners=codeowners,
            available_labels=labels,
        )

    async def get_pull_request(self, owner: str, repo: str, pr_number: int) -> PullRequestData:
        self._check_repo_allowed(owner, repo)
        raw = await self._get_json(f"/repos/{owner}/{repo}/pulls/{pr_number}")
        pr = PullRequestData.model_validate(
            {**raw, "base_ref": raw.get("base", {}).get("ref", "main"),
             "head_ref": raw.get("head", {}).get("ref", "")}
        )
        head_sha = raw.get("head", {}).get("sha", "")
        files, commits, reviews, checks = await asyncio.gather(
            self._get_json(f"/repos/{owner}/{repo}/pulls/{pr_number}/files",
                           params={"per_page": 100}),
            self._get_json(f"/repos/{owner}/{repo}/pulls/{pr_number}/commits",
                           params={"per_page": 100}),
            self._get_json(f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews"),
            self._get_check_runs(owner, repo, head_sha) if head_sha else asyncio.sleep(0, []),
        )
        pr.changed_files = [FileChange.model_validate(f) for f in files]
        pr.commits = [
            CommitInfo(sha=c.get("sha", ""), message=c.get("commit", {}).get("message", ""))
            for c in commits
        ]
        pr.reviews = [Review.model_validate(r) for r in reviews]
        pr.check_runs = checks
        pr.linked_issues = extract_linked_issues(pr.body or "")
        return pr

    async def _get_check_runs(self, owner: str, repo: str, sha: str) -> list[CheckRun]:
        try:
            raw = await self._get_json(f"/repos/{owner}/{repo}/commits/{sha}/check-runs")
        except GitHubNotFoundError:
            return []
        return [CheckRun.model_validate(run) for run in raw.get("check_runs", [])]

    async def is_assignable(self, owner: str, repo: str, username: str) -> bool:
        """True if the user can be assigned issues in this repository."""
        self._check_repo_allowed(owner, repo)
        try:
            response = await self._request(
                "GET", f"/repos/{owner}/{repo}/assignees/{username}"
            )
        except GitHubNotFoundError:
            return False
        return response.status_code == 204

    # ------------------------------------------------------------- mutations

    async def add_labels(
        self, owner: str, repo: str, issue_number: int, labels: list[str]
    ) -> list[Label]:
        self._check_repo_allowed(owner, repo)
        self._check_mutations("add_labels")
        raw = (
            await self._request(
                "POST",
                f"/repos/{owner}/{repo}/issues/{issue_number}/labels",
                json={"labels": labels},
            )
        ).json()
        return [Label.model_validate(label) for label in raw]

    async def post_comment(
        self, owner: str, repo: str, issue_number: int, body: str
    ) -> Comment:
        self._check_repo_allowed(owner, repo)
        self._check_mutations("post_comment")
        raw = (
            await self._request(
                "POST",
                f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
                json={"body": body},
            )
        ).json()
        return Comment.model_validate(raw)

    async def assign_issue(
        self, owner: str, repo: str, issue_number: int, assignees: list[str]
    ) -> IssueData:
        self._check_repo_allowed(owner, repo)
        self._check_mutations("assign_issue")
        raw = (
            await self._request(
                "POST",
                f"/repos/{owner}/{repo}/issues/{issue_number}/assignees",
                json={"assignees": assignees},
            )
        ).json()
        raw["is_pull_request"] = "pull_request" in raw
        return IssueData.model_validate(raw)

    async def close_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        *,
        state_reason: str = "completed",  # completed | not_planned
    ) -> IssueData:
        self._check_repo_allowed(owner, repo)
        self._check_mutations("close_issue")
        raw = (
            await self._request(
                "PATCH",
                f"/repos/{owner}/{repo}/issues/{issue_number}",
                json={"state": "closed", "state_reason": state_reason},
            )
        ).json()
        raw["is_pull_request"] = "pull_request" in raw
        return IssueData.model_validate(raw)

    async def reopen_issue(self, owner: str, repo: str, issue_number: int) -> IssueData:
        self._check_repo_allowed(owner, repo)
        self._check_mutations("reopen_issue")
        raw = (
            await self._request(
                "PATCH",
                f"/repos/{owner}/{repo}/issues/{issue_number}",
                json={"state": "open"},
            )
        ).json()
        raw["is_pull_request"] = "pull_request" in raw
        return IssueData.model_validate(raw)

    async def request_review(
        self, owner: str, repo: str, pr_number: int, reviewers: list[str]
    ) -> None:
        self._check_repo_allowed(owner, repo)
        self._check_mutations("request_review")
        await self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/requested_reviewers",
            json={"reviewers": reviewers},
        )


def extract_linked_issues(body: str) -> list[int]:
    """Parse 'closes #N' / 'fixes #N' / 'resolves #N' references from a PR body."""
    return sorted({int(m) for m in _CLOSING_KEYWORDS.findall(body)})
