from __future__ import annotations

from typing import Any, cast

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from fakes import FakeGitHubClient, FakeLLM, make_issue
from test_classification import bug_assessment
from test_duplicates import items
from triage_agent.agent.deps import TriageDeps
from triage_agent.agent.graph import build_graph
from triage_agent.agent.state import (
    ActionKind,
    ActionRisk,
    ApprovalDecision,
    DuplicateVerdict,
    ExecutionStatus,
    ProposedAction,
    TriageState,
)
from triage_agent.config import AppConfig
from triage_agent.github.models import SearchResult
from triage_agent.policies.models import RepositoryPolicy
from triage_agent.safety.approvals import (
    apply_decisions,
    parse_resume_payload,
    render_approval_request,
)
from triage_agent.safety.guardrails import SafetyGuardrails


def action(kind: ActionKind, risk: ActionRisk, **params: Any) -> ProposedAction:
    return ProposedAction(kind=kind, params=params, reason="test", risk=risk)


def state_with(
    actions: list[ProposedAction], *, confidence: float = 0.95, security: bool = False
) -> TriageState:
    return TriageState(
        owner="acme",
        repo="backend",
        issue_number=42,
        issue_data=make_issue(42),
        issue_assessment=bug_assessment(
            confidence=confidence, is_security_sensitive=security
        ),
        proposed_actions=actions,
    )


LABELS = action(ActionKind.ADD_LABELS, ActionRisk.SAFE, labels=["bug"])
COMMENT = action(ActionKind.POST_COMMENT, ActionRisk.LOW, comment="hi", marker="<!-- m -->")
ASSIGN = action(ActionKind.ASSIGN_ISSUE, ActionRisk.HIGH, assignees=["alice"])
CLOSE = action(ActionKind.CLOSE_ISSUE, ActionRisk.DESTRUCTIVE, state_reason="not_planned")

PERMISSIVE = RepositoryPolicy(
    automatic_labels=True, automatic_comments=True, automatic_assignment=True
)


# ------------------------------------------------------------------ guardrails


def test_destructive_never_auto_approved_even_at_full_confidence() -> None:
    guardrails = SafetyGuardrails(AppConfig())
    result = guardrails.evaluate(state_with([CLOSE], confidence=1.0), PERMISSIVE)
    assert result.auto_approved == []
    assert result.pending_approval == [CLOSE]


def test_safe_actions_auto_approved_at_high_confidence() -> None:
    guardrails = SafetyGuardrails(AppConfig())
    result = guardrails.evaluate(
        state_with([LABELS, COMMENT, ASSIGN], confidence=0.95), PERMISSIVE
    )
    assert {a.kind for a in result.auto_approved} == {
        ActionKind.ADD_LABELS,
        ActionKind.POST_COMMENT,
        ActionKind.ASSIGN_ISSUE,
    }
    assert all(a.requires_approval is False for a in result.auto_approved)


def test_mid_confidence_band_allows_safe_but_holds_riskier() -> None:
    guardrails = SafetyGuardrails(AppConfig())
    result = guardrails.evaluate(
        state_with([LABELS, COMMENT, ASSIGN], confidence=0.80), PERMISSIVE
    )
    assert [a.kind for a in result.auto_approved] == [ActionKind.ADD_LABELS]
    assert {a.kind for a in result.pending_approval} == {
        ActionKind.POST_COMMENT,
        ActionKind.ASSIGN_ISSUE,
    }


def test_below_review_threshold_nothing_runs() -> None:
    guardrails = SafetyGuardrails(AppConfig())
    result = guardrails.evaluate(
        state_with([LABELS, COMMENT], confidence=0.5), PERMISSIVE
    )
    assert result.auto_approved == []
    assert len(result.pending_approval) == 2


def test_policy_veto_overrides_confidence() -> None:
    guardrails = SafetyGuardrails(AppConfig())
    restrictive = RepositoryPolicy(automatic_labels=False)
    result = guardrails.evaluate(state_with([LABELS], confidence=0.99), restrictive)
    assert result.auto_approved == []
    assert "policy" in result.escalation_reasons[0]


def test_security_context_holds_comments_and_people_actions() -> None:
    guardrails = SafetyGuardrails(AppConfig())
    result = guardrails.evaluate(
        state_with([LABELS, COMMENT, ASSIGN], confidence=0.97, security=True),
        PERMISSIVE,
    )
    assert [a.kind for a in result.auto_approved] == [ActionKind.ADD_LABELS]
    assert {a.kind for a in result.pending_approval} == {
        ActionKind.POST_COMMENT,
        ActionKind.ASSIGN_ISSUE,
    }


def test_bulk_change_holds_everything() -> None:
    guardrails = SafetyGuardrails(AppConfig())
    many = [
        action(ActionKind.ADD_LABELS, ActionRisk.SAFE, labels=[f"l{i}"])
        for i in range(6)
    ]
    result = guardrails.evaluate(state_with(many, confidence=0.99), PERMISSIVE)
    assert result.auto_approved == []
    assert len(result.pending_approval) == 6


# ------------------------------------------------------------------- approvals


def test_parse_resume_payload_variants() -> None:
    pending = [LABELS, CLOSE]
    assert all(d.approved for d in parse_resume_payload(True, pending))
    assert not any(d.approved for d in parse_resume_payload("no", pending))
    decisions = parse_resume_payload(
        [{"action_index": 1, "approved": True}], pending
    )
    assert decisions[0].action_index == 1


def test_apply_decisions_defaults_to_rejection() -> None:
    pending = [LABELS, CLOSE]
    approved, rejected = apply_decisions(
        pending, [ApprovalDecision(action_index=0, approved=True)]
    )
    assert [a.kind for a in approved] == [ActionKind.ADD_LABELS]
    assert [a.kind for a in rejected] == [ActionKind.CLOSE_ISSUE]  # silence != consent


def test_render_approval_request_format() -> None:
    text = render_approval_request(state_with([CLOSE]), [CLOSE])
    assert "PROPOSED ACTION" in text
    assert "Close as not_planned" in text
    assert "Risk: destructive" in text
    assert "Approve? [yes/no]" in text


# ------------------------------------------- end-to-end critical invariant


def duplicate_scenario_deps(*, dry_run: bool, approval_mode: str) -> tuple[TriageDeps, FakeGitHubClient]:
    llm = FakeLLM()
    llm.queue(bug_assessment())
    llm.queue(DuplicateVerdict(duplicate_of=7, confidence=0.96, reason="same exception"))
    fake = FakeGitHubClient()
    fake.issues[42] = make_issue(42)
    fake.search_results = SearchResult(total_count=3, items=items())
    deps = TriageDeps(
        github=cast(Any, fake),
        llm=cast(Any, llm),
        dry_run=dry_run,
        approval_mode=approval_mode,  # type: ignore[arg-type]
    )
    return deps, fake


async def test_critical_invariant_close_never_executes_without_confirmation() -> None:
    """THE safety invariant: a destructive action must never reach GitHub
    without a human confirmation — even live, even at 96% confidence."""
    deps, fake = duplicate_scenario_deps(dry_run=False, approval_mode="none")
    result = await build_graph(deps).ainvoke(
        TriageState(owner="acme", repo="backend", issue_number=42)
    )
    state = TriageState.model_validate(result)

    proposed_kinds = {a.kind for a in state.proposed_actions}
    assert ActionKind.CLOSE_ISSUE in proposed_kinds  # it WAS proposed...
    assert "close_issue" not in fake.mutation_calls()  # ...but never executed
    assert any(a.kind is ActionKind.CLOSE_ISSUE for a in state.pending_approval)
    assert state.escalated_to_human is True
    # Safe action (labels) ran automatically at 93% confidence.
    assert "add_labels" in fake.mutation_calls()


async def test_interrupt_flow_approval_executes_close() -> None:
    deps, fake = duplicate_scenario_deps(dry_run=False, approval_mode="interrupt")
    graph = build_graph(deps, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "approval-1"}}

    first = await graph.ainvoke(
        TriageState(owner="acme", repo="backend", issue_number=42), config
    )
    # Graph paused at the approval interrupt with a rendered request.
    assert "__interrupt__" in first
    payload = first["__interrupt__"][0].value
    assert "PROPOSED ACTION" in payload["request"]
    assert "close_issue" not in fake.mutation_calls()  # nothing destructive yet

    result = await graph.ainvoke(Command(resume=True), config)
    state = TriageState.model_validate(result)
    assert "close_issue" in fake.mutation_calls()
    executed_kinds = {
        e.action.kind: e.status for e in state.executed_actions
    }
    assert executed_kinds[ActionKind.CLOSE_ISSUE] is ExecutionStatus.OK


async def test_interrupt_flow_denial_blocks_close() -> None:
    deps, fake = duplicate_scenario_deps(dry_run=False, approval_mode="interrupt")
    graph = build_graph(deps, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "approval-2"}}

    first = await graph.ainvoke(
        TriageState(owner="acme", repo="backend", issue_number=42), config
    )
    assert "__interrupt__" in first

    result = await graph.ainvoke(Command(resume=False), config)
    state = TriageState.model_validate(result)
    assert "close_issue" not in fake.mutation_calls()
    blocked = [
        e for e in state.executed_actions if e.status is ExecutionStatus.BLOCKED
    ]
    assert any(e.action.kind is ActionKind.CLOSE_ISSUE for e in blocked)
