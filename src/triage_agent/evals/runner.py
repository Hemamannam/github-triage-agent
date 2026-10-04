"""Evaluation runner: execute every scenario through the real pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import structlog

from triage_agent.agent.deps import TriageDeps
from triage_agent.evals.metrics import EvalMetrics, compute_metrics, render_report
from triage_agent.evals.scenarios import (
    Scenario,
    ScenarioResult,
    ScriptedGitHub,
    ScriptedLLM,
    check_expectations,
    load_scenarios,
)
from triage_agent.runner import TriageRunner

logger = structlog.get_logger("evals.runner")

def _default_scenario_dir() -> Path:
    """Scenario directory for both source checkouts and installed packages."""
    candidates = [
        Path.cwd() / "evals" / "scenarios",
        Path(__file__).resolve().parents[3] / "evals" / "scenarios",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_SCENARIO_DIR = _default_scenario_dir()


async def run_scenario(scenario: Scenario, llm: Any | None = None) -> ScenarioResult:
    github = ScriptedGitHub(scenario)
    if llm is None:
        llm = ScriptedLLM(scenario.llm)
    deps = TriageDeps(
        github=cast(Any, github),
        llm=cast(Any, llm),
        config=scenario.app_config(),
        # Live-mode semantics with a scripted GitHub: what executes here is
        # exactly what would hit the real API, which is what safety metrics
        # must measure.
        dry_run=False,
        approval_mode="none",
    )
    runner = TriageRunner(deps)
    start = time.perf_counter()
    outcome = await runner.run("acme", "backend", scenario.issue_number)
    latency = time.perf_counter() - start
    final_status = "awaiting_approval" if outcome.awaiting_approval else (
        "failed" if outcome.state.error else "completed"
    )
    failures, unsafe = check_expectations(
        scenario, outcome.state, github.mutations, final_status
    )
    return ScenarioResult(
        scenario=scenario,
        state=outcome.state,
        mutations=github.mutations,
        final_status=final_status,
        latency_s=latency,
        failures=failures,
        unsafe_actions=unsafe,
    )


async def run_evaluation(
    scenario_dir: Path | None = None,
    *,
    json_output: Path | None = None,
    live: bool = False,
) -> tuple[EvalMetrics, list[ScenarioResult]]:
    """Run the suite. ``live=True`` swaps the scripted LLM for the real,
    configured backend — measuring live model judgment on the same scenarios
    (scripted-only scenarios such as simulated outages are skipped)."""
    directory = scenario_dir or DEFAULT_SCENARIO_DIR
    scenarios = load_scenarios(directory)
    if not scenarios:
        raise FileNotFoundError(f"no scenario files found in {directory}")

    live_llm: Any | None = None
    if live:
        from triage_agent.agent.llm import build_llm
        from triage_agent.config import Settings, load_app_config

        settings = Settings()
        app_config = load_app_config(settings.triage_config)
        live_llm = build_llm(settings, app_config.llm)
        if live_llm is None:
            raise RuntimeError(
                "live evaluation requires ANTHROPIC_API_KEY (or OPENAI_API_KEY "
                "with llm.provider=openai) to be set"
            )
        scenarios = [s for s in scenarios if s.live]
        logger.info(
            "eval.live_mode", model=app_config.llm.model, scenarios=len(scenarios)
        )

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        logger.info("eval.scenario", name=scenario.name)
        results.append(await run_scenario(scenario, llm=live_llm))

    metrics = compute_metrics(results)
    if json_output is not None:
        payload = {
            "metrics": metrics.to_dict(),
            "results": [
                {
                    "name": r.scenario.name,
                    "passed": r.passed,
                    "failures": r.failures,
                    "unsafe_actions": r.unsafe_actions,
                    "final_status": r.final_status,
                    "latency_ms": round(r.latency_s * 1000, 1),
                    "mutations": [name for name, _ in r.mutations],
                }
                for r in results
            ],
        }
        json_output.write_text(json.dumps(payload, indent=2))
    return metrics, results


def main() -> None:  # pragma: no cover - thin CLI shim
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Run the triage evaluation suite")
    parser.add_argument("--scenarios", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--live", action="store_true",
        help="use the real configured LLM instead of scripted outputs",
    )
    args = parser.parse_args()

    metrics, results = asyncio.run(
        run_evaluation(args.scenarios, json_output=args.json, live=args.live)
    )
    print(render_report(metrics, results))
    raise SystemExit(0 if metrics.passed == metrics.scenarios else 1)


if __name__ == "__main__":  # pragma: no cover
    main()
