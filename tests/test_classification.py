from __future__ import annotations

from typing import Any, cast

from fakes import FakeGitHubClient, FakeLLM, make_issue, make_pr
from triage_agent.agent.deps import TriageDeps
from triage_agent.agent.graph import build_graph
from triage_agent.agent.llm import LLMError
from triage_agent.agent.nodes import filter_labels, pr_signals
from triage_agent.agent.state import (
    Classification,
    IssueAssessment,
    PRAssessment,
    Priority,
    Severity,
    TriageState,
)
from triage_agent.github.models import CheckRun, FileChange


def bug_assessment(**overrides: Any) -> IssueAssessment:
    defaults: dict[str, Any] = {
        "classification": Classification.BUG,
        "confidence": 0.93,
        "severity": Severity.HIGH,
        "priority": Priority.HIGH,
        "suggested_labels": ["bug", "P1", "made-up-label"],
        "missing_information": [],
        "search_keywords": ["oauth", "token", "crash"],
        "reasoning_summary": "Reproducible crash in OAuth callback.",
    }
    defaults.update(overrides)
    return IssueAssessment.model_validate(defaults)


def make_deps(llm: FakeLLM | None) -> tuple[TriageDeps, FakeGitHubClient]:
    fake = FakeGitHubClient()
    deps = TriageDeps(github=cast(Any, fake), llm=cast(Any, llm), dry_run=True)
    return deps, fake


async def run_graph(deps: TriageDeps, number: int = 42) -> TriageState:
    graph = build_graph(deps)
    result = await graph.ainvoke(
        TriageState(owner="acme", repo="backend", issue_number=number)
    )
    return TriageState.model_validate(result)


# ------------------------------------------------------------- issue pipeline


async def test_issue_classified_with_structured_output() -> None:
    llm = FakeLLM()
    llm.queue(bug_assessment())
    deps, fake = make_deps(llm)
    fake.issues[42] = make_issue(42)

    state = await run_graph(deps)
    assert state.issue_assessment is not None
    assert state.issue_assessment.classification is Classification.BUG
    assert state.confidence == 0.93
    # Hallucinated label dropped; real labels kept with repo casing.
    assert state.issue_assessment.suggested_labels == ["bug", "P1"]
    # The classify prompt carried repository context.
    schema, _system, prompt = llm.calls[0]
    assert schema == "IssueAssessment"
    assert "acme/backend" in prompt
    assert "OAuth callback crashes" in prompt


async def test_llm_failure_degrades_to_escalation() -> None:
    llm = FakeLLM()
    llm.queue_error("IssueAssessment", LLMError("model unavailable"))
    deps, fake = make_deps(llm)
    fake.issues[42] = make_issue(42)

    state = await run_graph(deps)
    assert state.issue_assessment is None
    assert state.escalated_to_human is True
    assert "classification failed" in state.escalation_reason
    assert state.tool_errors
    assert state.final_summary  # run still completed


async def test_recent_activity_gathered_for_issues() -> None:
    from triage_agent.github.models import SearchIssueItem, SearchResult

    llm = FakeLLM()
    llm.queue(bug_assessment())
    deps, fake = make_deps(llm)
    fake.issues[42] = make_issue(42)
    fake.search_results = SearchResult(
        total_count=2,
        items=[
            SearchIssueItem(number=42, title="self"),
            SearchIssueItem(number=41, title="other recent issue"),
        ],
    )
    state = await run_graph(deps)
    # The item under triage is excluded from recent activity.
    assert [item.number for item in state.recent_activity] == [41]


# ---------------------------------------------------------------- PR pipeline


def big_failing_pr() -> Any:
    return make_pr(
        10,
        additions=900,
        deletions=100,
        changed_files=[
            FileChange(filename="backend/core.py", additions=900, deletions=100)
        ],
        check_runs=[CheckRun(name="ci", status="completed", conclusion="failure")],
    )


async def test_pr_deterministic_signals_override_llm() -> None:
    llm = FakeLLM()
    llm.queue(
        PRAssessment(
            classification=Classification.FEATURE_REQUEST,
            confidence=0.85,
            summary="Adds a rate limiter.",
            reasoning_summary="Feature PR.",
            ci_failing=False,        # model got these wrong on purpose
            unusually_large=False,
            missing_tests=False,
            suggested_labels=["enhancement"],
        )
    )
    deps, fake = make_deps(llm)
    fake.issues[10] = make_issue(10, is_pull_request=True)
    fake.prs[10] = big_failing_pr()

    state = await run_graph(deps, number=10)
    assert state.pr_assessment is not None
    assert state.pr_assessment.ci_failing is True          # deterministic override
    assert state.pr_assessment.unusually_large is True     # deterministic override
    assert state.pr_assessment.missing_tests is True       # code changed, no tests
    assert state.pr_assessment.suggested_labels == ["enhancement"]


def test_pr_signals_computation() -> None:
    pr = make_pr(
        11,
        changed_files=[
            FileChange(filename="src/app.py", additions=50),
            FileChange(filename="tests/test_app.py", additions=30),
            FileChange(filename="docs/guide.md", additions=10),
        ],
    )
    signals = pr_signals(pr)
    assert signals["touches_code"] is True
    assert signals["has_test_changes"] is True
    assert signals["has_doc_changes"] is True
    assert signals["missing_tests"] is False
    assert signals["unusually_large"] is False


def test_pr_signals_docs_only_pr_needs_no_tests() -> None:
    pr = make_pr(12, changed_files=[FileChange(filename="README.md", additions=5)])
    signals = pr_signals(pr)
    assert signals["touches_code"] is False
    assert signals["missing_tests"] is False


# -------------------------------------------------------------- label filter


def test_filter_labels_dedupes_and_respects_repo_casing() -> None:
    from fakes import make_context

    ctx = make_context()
    assert filter_labels(["BUG", "bug", "nonexistent", "p1"], ctx) == ["bug", "P1"]
    assert filter_labels(["bug"], None) == []
