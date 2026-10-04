"""Aggregate metrics over scenario results and render the report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from triage_agent.agent.state import ExecutionStatus
from triage_agent.evals.scenarios import ScenarioResult


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _fmt(value: float | None, *, pct: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" if pct else f"{value:.2f}"


@dataclass
class EvalMetrics:
    scenarios: int
    passed: int
    task_success_rate: float | None
    classification_accuracy: float | None
    correct_label_rate: float | None
    duplicate_precision: float | None
    duplicate_recall: float | None
    unsafe_action_rate: float | None
    escalation_accuracy: float | None
    tool_call_success_rate: float | None
    avg_retries: float
    avg_latency_s: float
    total_latency_s: float

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def compute_metrics(results: list[ScenarioResult]) -> EvalMetrics:
    n = len(results)
    passed = sum(1 for r in results if r.passed)

    # Classification accuracy over scenarios that state an expectation.
    cls_total = cls_ok = 0
    for r in results:
        expected = r.scenario.expected.classification
        if expected is None:
            continue
        cls_total += 1
        actual = r.state.classification.value if r.state.classification else None
        if actual == expected:
            cls_ok += 1

    # Label correctness over scenarios that constrain labels.
    label_total = label_ok = 0
    for r in results:
        exp = r.scenario.expected
        if not exp.required_labels and exp.acceptable_labels is None:
            continue
        label_total += 1
        if not any("label" in f for f in r.failures):
            label_ok += 1

    # Duplicate precision/recall over scenarios that state an expectation.
    tp = fp = fn = 0
    for r in results:
        expected_dup = r.scenario.expected.duplicate
        if expected_dup is None:
            continue
        detected = bool(
            r.state.duplicate_verdict and r.state.duplicate_verdict.duplicate_of
        )
        if detected and expected_dup:
            tp += 1
        elif detected and not expected_dup:
            fp += 1
        elif not detected and expected_dup:
            fn += 1

    # Escalation accuracy over scenarios that state an expectation.
    esc_total = esc_ok = 0
    for r in results:
        expected_esc = r.scenario.expected.requires_human_confirmation
        if expected_esc is None:
            continue
        esc_total += 1
        actual_esc = r.state.escalated_to_human or bool(r.state.pending_approval)
        if actual_esc == expected_esc:
            esc_ok += 1

    # Tool-call reliability over every executed action in every run.
    calls_total = calls_ok = 0
    retries = 0
    for r in results:
        for executed in r.state.executed_actions:
            if executed.status is ExecutionStatus.BLOCKED:
                continue  # never attempted, by design
            calls_total += 1
            retries += executed.retries
            if executed.status is not ExecutionStatus.FAILED:
                calls_ok += 1

    unsafe = sum(1 for r in results if r.unsafe_actions)
    total_latency = sum(r.latency_s for r in results)

    return EvalMetrics(
        scenarios=n,
        passed=passed,
        task_success_rate=_rate(passed, n),
        classification_accuracy=_rate(cls_ok, cls_total),
        correct_label_rate=_rate(label_ok, label_total),
        duplicate_precision=_rate(tp, tp + fp),
        duplicate_recall=_rate(tp, tp + fn),
        unsafe_action_rate=_rate(unsafe, n),
        escalation_accuracy=_rate(esc_ok, esc_total),
        tool_call_success_rate=_rate(calls_ok, calls_total),
        avg_retries=retries / calls_total if calls_total else 0.0,
        avg_latency_s=total_latency / n if n else 0.0,
        total_latency_s=total_latency,
    )


def render_report(metrics: EvalMetrics, results: list[ScenarioResult]) -> str:
    lines = [
        "Evaluation Results",
        "==================",
        f"Scenarios:                {metrics.scenarios}",
        f"Task success:             {_fmt(metrics.task_success_rate)} "
        f"({metrics.passed}/{metrics.scenarios})",
        f"Classification accuracy:  {_fmt(metrics.classification_accuracy)}",
        f"Correct-label rate:       {_fmt(metrics.correct_label_rate)}",
        f"Duplicate precision:      {_fmt(metrics.duplicate_precision)}",
        f"Duplicate recall:         {_fmt(metrics.duplicate_recall)}",
        f"Unsafe-action rate:       {_fmt(metrics.unsafe_action_rate)}",
        f"Human-escalation accuracy:{_fmt(metrics.escalation_accuracy)}",
        f"Tool-call success rate:   {_fmt(metrics.tool_call_success_rate)}",
        f"Average retries/call:     {metrics.avg_retries:.2f}",
        f"Mean triage latency:      {metrics.avg_latency_s * 1000:.0f} ms",
        f"Total wall time:          {metrics.total_latency_s:.2f} s",
    ]
    failed = [r for r in results if not r.passed]
    if failed:
        lines.append("")
        lines.append("Failed scenarios:")
        for r in failed:
            lines.append(f"  ✗ {r.scenario.name}")
            for failure in r.failures:
                lines.append(f"      - {failure}")
            for unsafe in r.unsafe_actions:
                lines.append(f"      - UNSAFE ACTION: {unsafe}")
    else:
        lines.append("")
        lines.append("All scenarios passed.")
    return "\n".join(lines)
