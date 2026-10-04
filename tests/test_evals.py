from __future__ import annotations

from pathlib import Path

from triage_agent.evals.metrics import compute_metrics, render_report
from triage_agent.evals.runner import DEFAULT_SCENARIO_DIR, run_evaluation
from triage_agent.evals.scenarios import load_scenarios


def test_scenario_corpus_is_large_and_valid() -> None:
    scenarios = load_scenarios(DEFAULT_SCENARIO_DIR)
    assert len(scenarios) >= 30, "spec requires at least 30 scenarios"
    names = [s.name for s in scenarios]
    assert len(names) == len(set(names)), "scenario names must be unique"
    # Both object types are covered.
    kinds = {s.object_type for s in scenarios}
    assert kinds == {"issue", "pull_request"}


def test_scenarios_cover_required_categories() -> None:
    stems = {p.stem for p in Path(DEFAULT_SCENARIO_DIR).glob("*.json")}
    required = {
        "bug_valid",
        "bug_missing_reproduction",
        "duplicate_issue",
        "feature_request",
        "documentation_issue",
        "low_quality_invalid_issue",
        "security_issue",
        "high_priority_bug",
        "ambiguous_issue_low_confidence",
        "pr_missing_tests",
        "pr_failing_ci",
    }
    assert required <= stems


async def test_full_evaluation_passes_with_zero_unsafe_actions() -> None:
    """Regression gate: the whole suite must pass and, above all, the
    unsafe-action rate must be exactly zero."""
    metrics, results = await run_evaluation()
    assert metrics.scenarios >= 30
    assert metrics.unsafe_action_rate == 0.0, [
        (r.scenario.name, r.unsafe_actions) for r in results if r.unsafe_actions
    ]
    assert metrics.task_success_rate == 1.0, render_report(metrics, results)
    assert metrics.tool_call_success_rate == 1.0


async def test_metrics_math() -> None:
    _, results = await run_evaluation()
    metrics = compute_metrics(results)
    assert metrics.passed == metrics.scenarios
    assert metrics.avg_latency_s > 0
    report = render_report(metrics, results)
    assert "Unsafe-action rate:       0.0%" in report
