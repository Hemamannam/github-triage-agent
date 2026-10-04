from __future__ import annotations

import base64

import httpx
import pytest
import respx

from conftest import issue_payload
from triage_agent.github.client import (
    GitHubClient,
    GitHubNotFoundError,
    GitHubServerError,
    MutationBlockedError,
    RepositoryNotAllowedError,
    extract_linked_issues,
)

API = "https://api.github.com"


@respx.mock
async def test_get_issue_with_comments(gh_client: GitHubClient) -> None:
    respx.get(f"{API}/repos/acme/backend/issues/42").respond(
        json=issue_payload(42, comments=1)
    )
    respx.get(f"{API}/repos/acme/backend/issues/42/comments").respond(
        json=[{"id": 1, "user": {"login": "bob"}, "body": "same here"}]
    )
    issue = await gh_client.get_issue("acme", "backend", 42)
    assert issue.number == 42
    assert issue.comments_count == 1
    assert issue.comments[0].body == "same here"
    assert issue.label_names == ["bug"]
    assert issue.is_pull_request is False


@respx.mock
async def test_search_issues_scoped_to_repo(gh_client: GitHubClient) -> None:
    route = respx.get(f"{API}/search/issues").respond(
        json={"total_count": 1, "items": [issue_payload(7)]}
    )
    result = await gh_client.search_issues("acme", "backend", "login crash")
    assert result.total_count == 1
    assert result.items[0].number == 7
    sent_query = route.calls.last.request.url.params["q"]
    assert sent_query.startswith("repo:acme/backend ")


@respx.mock
async def test_retries_on_server_error_then_succeeds(gh_client: GitHubClient) -> None:
    route = respx.get(f"{API}/repos/acme/backend/issues/1")
    route.side_effect = [
        httpx.Response(500, json={"message": "boom"}),
        httpx.Response(200, json=issue_payload(1)),
    ]
    issue = await gh_client.get_issue("acme", "backend", 1, with_comments=False)
    assert issue.number == 1
    assert gh_client.metrics.retries == 1
    assert gh_client.metrics.last_call_retries == 1


@respx.mock
async def test_rate_limit_honoured_and_counted(gh_client: GitHubClient) -> None:
    route = respx.get(f"{API}/repos/acme/backend/issues/1")
    route.side_effect = [
        httpx.Response(
            429, json={"message": "API rate limit exceeded"}, headers={"Retry-After": "0"}
        ),
        httpx.Response(200, json=issue_payload(1)),
    ]
    issue = await gh_client.get_issue("acme", "backend", 1, with_comments=False)
    assert issue.number == 1
    assert gh_client.metrics.rate_limit_hits == 1


@respx.mock
async def test_retries_exhausted_raises(gh_client: GitHubClient) -> None:
    respx.get(f"{API}/repos/acme/backend/issues/1").respond(
        status_code=500, json={"message": "down"}
    )
    with pytest.raises(GitHubServerError):
        await gh_client.get_issue("acme", "backend", 1, with_comments=False)
    assert gh_client.metrics.failures == 1


@respx.mock
async def test_404_is_not_retried(gh_client: GitHubClient) -> None:
    route = respx.get(f"{API}/repos/acme/backend/issues/999").respond(
        status_code=404, json={"message": "Not Found"}
    )
    with pytest.raises(GitHubNotFoundError):
        await gh_client.get_issue("acme", "backend", 999, with_comments=False)
    assert route.call_count == 1


async def test_repo_allowlist_blocks_reads_and_writes(
    allowlisted_client: GitHubClient,
) -> None:
    with pytest.raises(RepositoryNotAllowedError):
        await allowlisted_client.get_issue("evil", "repo", 1)
    with pytest.raises(RepositoryNotAllowedError):
        await allowlisted_client.add_labels("evil", "repo", 1, ["bug"])


async def test_readonly_client_blocks_every_mutation(
    readonly_client: GitHubClient,
) -> None:
    for coro in (
        readonly_client.add_labels("acme", "backend", 1, ["bug"]),
        readonly_client.post_comment("acme", "backend", 1, "hi"),
        readonly_client.assign_issue("acme", "backend", 1, ["alice"]),
        readonly_client.close_issue("acme", "backend", 1),
        readonly_client.reopen_issue("acme", "backend", 1),
        readonly_client.request_review("acme", "backend", 1, ["alice"]),
    ):
        with pytest.raises(MutationBlockedError):
            await coro


@respx.mock
async def test_repository_context_gathers_everything(gh_client: GitHubClient) -> None:
    def b64(text: str) -> dict:
        return {
            "encoding": "base64",
            "content": base64.b64encode(text.encode()).decode(),
        }

    respx.get(f"{API}/repos/acme/backend").respond(
        json={
            "full_name": "acme/backend",
            "description": "Backend service",
            "topics": ["api"],
            "default_branch": "main",
        }
    )
    respx.get(f"{API}/repos/acme/backend/readme").respond(json=b64("# Backend"))
    respx.get(f"{API}/repos/acme/backend/contents/CONTRIBUTING.md").respond(
        json=b64("Please add tests.")
    )
    respx.get(f"{API}/repos/acme/backend/contents/.github/CODEOWNERS").respond(
        json=b64("/backend/ @backend-team")
    )
    respx.get(f"{API}/repos/acme/backend/labels").respond(
        json=[{"name": "bug"}, {"name": "P1"}]
    )
    ctx = await gh_client.get_repository_context("acme", "backend")
    assert ctx.info.full_name == "acme/backend"
    assert ctx.readme == "# Backend"
    assert ctx.contributing == "Please add tests."
    assert ctx.codeowners is not None and "@backend-team" in ctx.codeowners
    assert ctx.available_label_names == ["bug", "P1"]


@respx.mock
async def test_get_pull_request_composite(gh_client: GitHubClient) -> None:
    respx.get(f"{API}/repos/acme/backend/pulls/10").respond(
        json={
            "number": 10,
            "title": "Add rate limiter",
            "body": "Fixes #42",
            "state": "open",
            "user": {"login": "carol"},
            # GitHub sends *counts* under these names in the raw payload;
            # regression coverage for the count/object field collision.
            "changed_files": 2,
            "commits": 1,
            "comments": 3,
            "review_comments": 0,
            "additions": 120,
            "deletions": 5,
            "base": {"ref": "main"},
            "head": {"ref": "feat/rl", "sha": "abc123"},
        }
    )
    respx.get(f"{API}/repos/acme/backend/pulls/10/files").respond(
        json=[
            {"filename": "backend/limiter.py", "status": "added", "additions": 100},
            {"filename": "backend/app.py", "status": "modified", "additions": 20},
        ]
    )
    respx.get(f"{API}/repos/acme/backend/pulls/10/commits").respond(
        json=[{"sha": "abc123", "commit": {"message": "add limiter"}}]
    )
    respx.get(f"{API}/repos/acme/backend/pulls/10/reviews").respond(json=[])
    respx.get(f"{API}/repos/acme/backend/commits/abc123/check-runs").respond(
        json={
            "check_runs": [
                {"name": "ci", "status": "completed", "conclusion": "failure"}
            ]
        }
    )
    pr = await gh_client.get_pull_request("acme", "backend", 10)
    assert pr.number == 10
    assert [f.filename for f in pr.changed_files] == [
        "backend/limiter.py",
        "backend/app.py",
    ]
    assert pr.linked_issues == [42]
    assert pr.ci_failing is True
    assert pr.total_diff_size == 125


@respx.mock
async def test_mutations_hit_correct_endpoints(gh_client: GitHubClient) -> None:
    label_route = respx.post(f"{API}/repos/acme/backend/issues/1/labels").respond(
        json=[{"name": "bug"}]
    )
    close_route = respx.patch(f"{API}/repos/acme/backend/issues/1").respond(
        json=issue_payload(1, state="closed")
    )
    labels = await gh_client.add_labels("acme", "backend", 1, ["bug"])
    assert labels[0].name == "bug"
    assert label_route.called

    issue = await gh_client.close_issue("acme", "backend", 1, state_reason="not_planned")
    assert issue.state.value == "closed"
    import json

    sent = json.loads(close_route.calls.last.request.content)
    assert sent == {"state": "closed", "state_reason": "not_planned"}


def test_extract_linked_issues() -> None:
    body = "Fixes #42, closes #7 and resolves #42. Related to #99 (not closing)."
    assert extract_linked_issues(body) == [7, 42]
    assert extract_linked_issues("no links") == []
