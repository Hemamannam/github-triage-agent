from __future__ import annotations

from typing import Any, cast

from fakes import FakeGitHubClient, FakeLLM, make_issue
from test_classification import bug_assessment
from triage_agent.agent.deps import TriageDeps
from triage_agent.agent.duplicates import (
    extract_keywords,
    lexical_similarity,
    score_candidates,
)
from triage_agent.agent.graph import build_graph
from triage_agent.agent.state import DuplicateVerdict, TriageState
from triage_agent.github.models import SearchIssueItem, SearchResult


def items() -> list[SearchIssueItem]:
    return [
        SearchIssueItem(
            number=7,
            title="OAuth callback crashes when token expires",
            body="KeyError in refresh_token when the token is expired",
        ),
        SearchIssueItem(number=8, title="Add dark mode support", body="UI request"),
        SearchIssueItem(number=42, title="self match", body="should be excluded"),
        SearchIssueItem(number=9, title="OAuth docs unclear", body="", is_pull_request=True),
    ]


def test_extract_keywords_drops_stopwords() -> None:
    kws = extract_keywords("The OAuth callback crashes when the token expires")
    assert "the" not in kws and "when" not in kws
    assert "oauth" in kws and "crashes" in kws


def test_lexical_similarity_orders_sensibly() -> None:
    issue = make_issue(42)
    sims = {item.number: lexical_similarity(issue, item) for item in items()[:2]}
    assert sims[7] > sims[8]
    assert sims[7] > 0.5


def test_score_candidates_excludes_self_prs_and_weak_hits() -> None:
    issue = make_issue(42)
    candidates = score_candidates(issue, items(), max_candidates=5)
    numbers = [c.issue_number for c in candidates]
    assert 42 not in numbers  # self
    assert 9 not in numbers   # PR
    assert numbers[0] == 7    # strongest first


async def test_duplicate_pipeline_with_llm_verdict() -> None:
    llm = FakeLLM()
    llm.queue(bug_assessment())
    llm.queue(
        DuplicateVerdict(duplicate_of=7, confidence=0.96, reason="same exception path")
    )
    fake = FakeGitHubClient()
    fake.issues[42] = make_issue(42)
    fake.search_results = SearchResult(total_count=4, items=items())
    deps = TriageDeps(github=cast(Any, fake), llm=cast(Any, llm), dry_run=True)

    result = await build_graph(deps).ainvoke(
        TriageState(owner="acme", repo="backend", issue_number=42)
    )
    state = TriageState.model_validate(result)
    assert state.duplicate_verdict is not None
    assert state.duplicate_verdict.duplicate_of == 7
    dup = next(c for c in state.duplicate_candidates if c.issue_number == 7)
    assert dup.is_duplicate is True
    assert dup.llm_confidence == 0.96
    # The duplicate search query came from the assessment's keywords.
    search_queries = [args[2] for name, args in fake.calls if name == "search_issues"]
    assert any("oauth" in q for q in search_queries)


async def test_hallucinated_duplicate_reference_is_discarded() -> None:
    llm = FakeLLM()
    llm.queue(bug_assessment())
    llm.queue(
        DuplicateVerdict(duplicate_of=9999, confidence=0.95, reason="made up")
    )
    fake = FakeGitHubClient()
    fake.issues[42] = make_issue(42)
    fake.search_results = SearchResult(total_count=4, items=items())
    deps = TriageDeps(github=cast(Any, fake), llm=cast(Any, llm), dry_run=True)

    result = await build_graph(deps).ainvoke(
        TriageState(owner="acme", repo="backend", issue_number=42)
    )
    state = TriageState.model_validate(result)
    assert state.duplicate_verdict is not None
    assert state.duplicate_verdict.duplicate_of is None
    assert state.duplicate_verdict.confidence == 0.0
    assert "not" in state.duplicate_verdict.reason


async def test_codeowner_suggestions_for_issue_and_security_escalation() -> None:
    from triage_agent.agent.state import Classification, Priority, Severity

    llm = FakeLLM()
    llm.queue(
        bug_assessment(
            classification=Classification.SECURITY,
            severity=Severity.MEDIUM,
            priority=Priority.MEDIUM,
            suggested_labels=["bug"],
            search_keywords=["backend", "auth"],
        )
    )
    fake = FakeGitHubClient()
    fake.issues[42] = make_issue(42)
    deps = TriageDeps(github=cast(Any, fake), llm=cast(Any, llm), dry_run=True)

    result = await build_graph(deps).ainvoke(
        TriageState(owner="acme", repo="backend", issue_number=42)
    )
    state = TriageState.model_validate(result)
    assert state.issue_assessment is not None
    # Security reports are force-escalated regardless of the model's rating.
    assert state.issue_assessment.priority is Priority.CRITICAL
    assert state.issue_assessment.severity in (Severity.HIGH, Severity.CRITICAL)
    # CODEOWNERS-grounded suggestion (fake context maps /backend/ to a team).
    assert "@backend-team" in state.codeowner_suggestions
