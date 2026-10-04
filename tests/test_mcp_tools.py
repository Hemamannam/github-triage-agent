from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from fakes import FakeGitHubClient, make_issue
from triage_agent.github.models import SearchIssueItem, SearchResult
from triage_agent.mcp.tools import TriageToolService


def make_service(*, dry_run: bool) -> tuple[TriageToolService, FakeGitHubClient]:
    fake = FakeGitHubClient()
    service = TriageToolService(cast(Any, fake), dry_run=dry_run)
    return service, fake


# ------------------------------------------------------------- input validation


async def test_invalid_owner_rejected() -> None:
    service, _ = make_service(dry_run=True)
    with pytest.raises(ValidationError):
        await service.get_issue("../etc/passwd", "repo", 1)


async def test_invalid_issue_number_rejected() -> None:
    service, _ = make_service(dry_run=True)
    with pytest.raises(ValidationError):
        await service.get_issue("acme", "backend", 0)


async def test_bad_label_rejected() -> None:
    service, fake = make_service(dry_run=False)
    with pytest.raises((ValidationError, ValueError)):
        await service.add_labels("acme", "backend", 1, ["x" * 100])
    assert fake.mutation_calls() == []


async def test_oversized_comment_rejected() -> None:
    service, fake = make_service(dry_run=False)
    with pytest.raises(ValidationError):
        await service.post_comment("acme", "backend", 1, "a" * 70_000)
    assert fake.mutation_calls() == []


async def test_invalid_assignee_username_rejected() -> None:
    service, fake = make_service(dry_run=False)
    with pytest.raises((ValidationError, ValueError)):
        await service.assign_issue("acme", "backend", 1, ["not a user!"])
    assert fake.mutation_calls() == []


# ------------------------------------------------------------------ read tools


async def test_get_issue_returns_payload() -> None:
    service, fake = make_service(dry_run=True)
    fake.issues[42] = make_issue(42)
    payload = await service.get_issue("acme", "backend", 42)
    assert payload["number"] == 42
    assert payload["repository"] == "acme/backend"


async def test_search_issues() -> None:
    service, fake = make_service(dry_run=True)
    fake.search_results = SearchResult(
        total_count=1,
        items=[SearchIssueItem(number=7, title="OAuth crash")],
    )
    payload = await service.search_issues("acme", "backend", "oauth crash")
    assert payload["total_count"] == 1


async def test_repository_context_truncates_readme() -> None:
    service, fake = make_service(dry_run=True)
    fake.context = fake.context.model_copy(update={"readme": "x" * 10_000})
    payload = await service.get_repository_context("acme", "backend")
    assert len(payload["readme"]) < 10_000
    assert "truncated" in payload["readme"]
    assert payload["codeowners"] is not None


# --------------------------------------------------------------------- dry run


async def test_dry_run_blocks_all_mutations() -> None:
    service, fake = make_service(dry_run=True)
    r1 = await service.add_labels("acme", "backend", 1, ["bug"])
    r2 = await service.post_comment("acme", "backend", 1, "hello")
    r3 = await service.assign_issue("acme", "backend", 1, ["alice"])
    r4 = await service.close_issue("acme", "backend", 1, "dup", confirm=True)
    assert {r["status"] for r in (r1, r2, r3, r4)} == {"dry_run"}
    assert all("would_execute" in r for r in (r1, r2, r3, r4))
    assert fake.mutation_calls() == []


# ------------------------------------------------------------------- live mode


async def test_live_mutations_execute() -> None:
    service, fake = make_service(dry_run=False)
    result = await service.add_labels("acme", "backend", 1, ["bug", "P1"])
    assert result == {"status": "ok", "labels": ["bug", "P1"]}
    assert fake.mutation_calls() == ["add_labels"]


async def test_assign_rejects_unverified_users_even_live() -> None:
    service, fake = make_service(dry_run=False)
    result = await service.assign_issue("acme", "backend", 1, ["ghost-user"])
    assert result["status"] == "rejected"
    assert fake.mutation_calls() == []


# --------------------------------------------- destructive confirmation gating


async def test_close_issue_without_confirm_never_mutates() -> None:
    """CRITICAL INVARIANT: a destructive action must never execute without
    explicit confirmation — in any mode."""
    for dry_run in (True, False):
        service, fake = make_service(dry_run=dry_run)
        result = await service.close_issue(
            "acme", "backend", 42, reason="duplicate of #7", confirm=False
        )
        assert result["status"] == "confirmation_required"
        assert result["proposed_action"]["action"] == "close_issue"
        assert fake.mutation_calls() == [], f"mutated despite confirm=False (dry_run={dry_run})"


async def test_close_issue_with_confirm_executes_live() -> None:
    service, fake = make_service(dry_run=False)
    result = await service.close_issue(
        "acme", "backend", 42, reason="duplicate of #7",
        state_reason="not_planned", confirm=True,
    )
    assert result["status"] == "ok"
    assert fake.mutation_calls() == ["close_issue"]
    assert fake.calls[-1] == ("close_issue", ("acme", "backend", 42, "not_planned"))


async def test_reopen_issue_gated_the_same_way() -> None:
    service, fake = make_service(dry_run=False)
    pending = await service.reopen_issue("acme", "backend", 42, reason="not fixed")
    assert pending["status"] == "confirmation_required"
    assert fake.mutation_calls() == []

    done = await service.reopen_issue("acme", "backend", 42, reason="not fixed", confirm=True)
    assert done["status"] == "ok"
    assert fake.mutation_calls() == ["reopen_issue"]
