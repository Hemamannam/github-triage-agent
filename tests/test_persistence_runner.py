from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from langgraph.checkpoint.memory import MemorySaver

from fakes import FakeGitHubClient, FakeLLM, make_issue
from test_classification import bug_assessment
from test_duplicates import items
from triage_agent.agent.deps import TriageDeps
from triage_agent.agent.state import DuplicateVerdict, TriageState
from triage_agent.github.models import SearchResult
from triage_agent.persistence.repositories import SqliteAuditStore
from triage_agent.runner import TriageRunner


@pytest.fixture
async def store(tmp_path: Path) -> Any:
    s = await SqliteAuditStore.open(tmp_path / "audit.db")
    yield s
    await s.aclose()


def simple_deps(**kwargs: Any) -> tuple[TriageDeps, FakeGitHubClient]:
    llm = FakeLLM()
    llm.queue(bug_assessment())
    fake = FakeGitHubClient()
    fake.issues[42] = make_issue(42)
    deps = TriageDeps(github=cast(Any, fake), llm=cast(Any, llm), **kwargs)
    return deps, fake


async def test_store_roundtrip(store: SqliteAuditStore) -> None:
    state = TriageState(
        owner="acme", repo="backend", issue_number=42,
        issue_data=make_issue(42), issue_assessment=bug_assessment(),
    )
    await store.start_run("r1", "acme/backend", 42, "issue")
    await store.record_decision("r1", state)
    await store.complete_run("r1", "completed", "all good")

    report = await store.get_run_report("r1")
    assert report is not None
    assert report["run"]["final_status"] == "completed"
    assert report["decisions"][0]["classification"] == "bug"
    assert report["decisions"][0]["confidence"] == 0.93

    runs = await store.list_runs()
    assert len(runs) == 1 and runs[0]["id"] == "r1"


async def test_runner_records_full_audit_trail(store: SqliteAuditStore) -> None:
    deps, _fake = simple_deps(dry_run=True)
    runner = TriageRunner(deps, store)
    outcome = await runner.run("acme", "backend", 42)

    assert outcome.awaiting_approval is False
    report = await store.get_run_report(outcome.run_id)
    assert report is not None
    assert report["run"]["final_status"] == "completed"
    assert report["run"]["summary"]  # traceable summary persisted
    assert report["decisions"], "decision row missing"
    # dry-run label execution recorded as a tool execution
    tools = [row["tool"] for row in report["tool_executions"]]
    assert "add_labels" in tools
    assert all(row["success"] == 1 for row in report["tool_executions"])


async def test_runner_marks_failed_runs(store: SqliteAuditStore) -> None:
    class DownFake(FakeGitHubClient):
        async def get_repository_context(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("github down")

    deps = TriageDeps(github=cast(Any, DownFake()), llm=None, dry_run=True)
    runner = TriageRunner(deps, store)
    outcome = await runner.run("acme", "backend", 42)
    report = await store.get_run_report(outcome.run_id)
    assert report is not None
    assert report["run"]["final_status"] == "failed"


async def test_runner_interrupt_and_resume_records_approvals(
    store: SqliteAuditStore,
) -> None:
    llm = FakeLLM()
    llm.queue(bug_assessment())
    llm.queue(DuplicateVerdict(duplicate_of=7, confidence=0.97, reason="same crash"))
    fake = FakeGitHubClient()
    fake.issues[42] = make_issue(42)
    fake.search_results = SearchResult(total_count=3, items=items())
    deps = TriageDeps(
        github=cast(Any, fake), llm=cast(Any, llm),
        dry_run=False, approval_mode="interrupt",
    )
    runner = TriageRunner(deps, store, checkpointer=MemorySaver())

    outcome = await runner.run("acme", "backend", 42)
    assert outcome.awaiting_approval is True
    assert "PROPOSED ACTION" in outcome.approval_request
    report = await store.get_run_report(outcome.run_id)
    assert report is not None and report["run"]["final_status"] == "awaiting_approval"
    assert "close_issue" not in fake.mutation_calls()

    resumed = await runner.resume(outcome.run_id, True)
    assert resumed.awaiting_approval is False
    assert "close_issue" in fake.mutation_calls()

    report = await store.get_run_report(outcome.run_id)
    assert report is not None
    assert report["run"]["final_status"] == "completed"
    approvals = report["approvals"]
    assert approvals and all(row["approved"] == 1 for row in approvals)
    tools = [row["tool"] for row in report["tool_executions"]]
    assert "close_issue" in tools


async def test_runner_resume_denial_records_rejections(store: SqliteAuditStore) -> None:
    llm = FakeLLM()
    llm.queue(bug_assessment())
    llm.queue(DuplicateVerdict(duplicate_of=7, confidence=0.97, reason="same crash"))
    fake = FakeGitHubClient()
    fake.issues[42] = make_issue(42)
    fake.search_results = SearchResult(total_count=3, items=items())
    deps = TriageDeps(
        github=cast(Any, fake), llm=cast(Any, llm),
        dry_run=False, approval_mode="interrupt",
    )
    runner = TriageRunner(deps, store, checkpointer=MemorySaver())

    outcome = await runner.run("acme", "backend", 42)
    await runner.resume(outcome.run_id, False)
    assert "close_issue" not in fake.mutation_calls()

    report = await store.get_run_report(outcome.run_id)
    assert report is not None
    assert report["approvals"]
    assert all(row["approved"] == 0 for row in report["approvals"])


def test_cli_help_smoke() -> None:
    from typer.testing import CliRunner

    from triage_agent.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "triage" in result.output.lower()
