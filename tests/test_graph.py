from __future__ import annotations

from typing import Any, cast

from langgraph.checkpoint.memory import MemorySaver

from fakes import FakeGitHubClient, make_issue, make_pr
from triage_agent.agent.deps import TriageDeps
from triage_agent.agent.graph import build_graph
from triage_agent.agent.state import ObjectType, TriageState


def make_deps(fake: FakeGitHubClient | None = None) -> tuple[TriageDeps, FakeGitHubClient]:
    fake = fake or FakeGitHubClient()
    return TriageDeps(github=cast(Any, fake), llm=None, dry_run=True), fake


async def test_graph_compiles_with_expected_nodes() -> None:
    deps, _ = make_deps()
    graph = build_graph(deps)
    node_names = set(graph.get_graph().nodes)
    expected = {
        "fetch_repository_context",
        "fetch_item",
        "classify_request",
        "gather_additional_context",
        "search_duplicates",
        "assess_priority_and_severity",
        "determine_actions",
        "safety_check",
        "human_approval",
        "execute_actions",
        "verify_actions",
        "generate_summary",
    }
    assert expected <= node_names


async def test_issue_run_flows_end_to_end() -> None:
    deps, fake = make_deps()
    fake.issues[42] = make_issue(42)
    graph = build_graph(deps)
    result = await graph.ainvoke(
        TriageState(owner="acme", repo="backend", issue_number=42)
    )
    state = TriageState.model_validate(result)
    assert state.issue_data is not None and state.issue_data.number == 42
    assert state.repository_context is not None
    assert state.object_type is ObjectType.ISSUE
    # Without an LLM the run must degrade to an explicit human escalation.
    assert state.escalated_to_human is True
    assert "Triage run" in state.final_summary
    assert state.error is None


async def test_pr_detected_and_fetched() -> None:
    deps, fake = make_deps()
    fake.issues[10] = make_issue(10, is_pull_request=True)
    fake.prs[10] = make_pr(10)
    graph = build_graph(deps)
    result = await graph.ainvoke(
        TriageState(owner="acme", repo="backend", issue_number=10)
    )
    state = TriageState.model_validate(result)
    assert state.object_type is ObjectType.PULL_REQUEST
    assert state.pr_data is not None and state.pr_data.number == 10
    # PR path must not run the duplicate search.
    assert not any(name == "search_issues" for name, _ in fake.calls)


async def test_fetch_failure_short_circuits_to_summary() -> None:
    class ExplodingFake(FakeGitHubClient):
        async def get_issue(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("github is down")

    deps, _ = make_deps(ExplodingFake())
    graph = build_graph(deps)
    result = await graph.ainvoke(
        TriageState(owner="acme", repo="backend", issue_number=1)
    )
    state = TriageState.model_validate(result)
    assert state.error is not None and "github is down" in state.error
    assert "FAILED" in state.final_summary
    # Classification must not have run.
    assert state.issue_assessment is None


async def test_graph_checkpointing_persists_state() -> None:
    deps, fake = make_deps()
    fake.issues[42] = make_issue(42)
    graph = build_graph(deps, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t-1"}}
    await graph.ainvoke(
        TriageState(owner="acme", repo="backend", issue_number=42), config
    )
    snapshot = graph.get_state(config)
    persisted = TriageState.model_validate(snapshot.values)
    assert persisted.issue_data is not None
    assert persisted.final_summary
