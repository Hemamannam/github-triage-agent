"""Command-line interface.

    triage-agent run --repo owner/project --issue 142 --dry-run
    triage-agent runs
    triage-agent show-run <run_id>
    triage-agent serve-mcp
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

from triage_agent.agent.state import ExecutionStatus, ObjectType, TriageState
from triage_agent.config import Settings, load_app_config
from triage_agent.observability.logging import configure_logging

app = typer.Typer(help="Autonomous GitHub issue/PR triage agent.", no_args_is_help=True)
console = Console()

_STATUS_ICONS = {
    ExecutionStatus.OK: "[green]✓[/green]",
    ExecutionStatus.DRY_RUN: "[cyan]◦[/cyan]",
    ExecutionStatus.SKIPPED: "[yellow]∅[/yellow]",
    ExecutionStatus.FAILED: "[red]✗[/red]",
    ExecutionStatus.BLOCKED: "[red]⊘[/red]",
}


def _parse_repo(repo: str) -> tuple[str, str]:
    if "/" not in repo:
        console.print("[red]--repo must be owner/name[/red]")
        raise typer.Exit(2)
    owner, name = repo.split("/", 1)
    return owner, name


def _print_state(state: TriageState) -> None:
    kind = "Pull request" if state.object_type is ObjectType.PULL_REQUEST else "Issue"
    console.print(f"Type: [bold]{kind}[/bold]")
    if state.classification:
        console.print(
            f"Classification: [bold]{state.classification.value}[/bold]   "
            f"Confidence: {state.confidence:.0%}"
        )
    if state.issue_assessment:
        console.print(
            f"Severity: {state.issue_assessment.severity.value}   "
            f"Priority: {state.issue_assessment.priority.value}"
        )
        if state.issue_assessment.missing_information:
            console.print(
                "Missing info: "
                + ", ".join(state.issue_assessment.missing_information)
            )
    if state.pr_assessment:
        flags = [
            name
            for name, on in [
                ("missing tests", state.pr_assessment.missing_tests),
                ("missing docs", state.pr_assessment.missing_documentation),
                ("CI failing", state.pr_assessment.ci_failing),
                ("unusually large", state.pr_assessment.unusually_large),
                ("security-sensitive", state.pr_assessment.is_security_sensitive),
            ]
            if on
        ]
        if flags:
            console.print(f"Flags: [yellow]{', '.join(flags)}[/yellow]")
        console.print(f"Summary: {state.pr_assessment.summary}")
    if state.duplicate_verdict and state.duplicate_verdict.duplicate_of:
        verdict = state.duplicate_verdict
        candidate = next(
            (c for c in state.duplicate_candidates if c.issue_number == verdict.duplicate_of),
            None,
        )
        title = f' — "{candidate.title}"' if candidate else ""
        sim = f", similarity {candidate.similarity:.2f}" if candidate else ""
        console.print(
            f"Possible duplicate: [bold]#{verdict.duplicate_of}[/bold]{title}"
            f" (confidence {verdict.confidence:.0%}{sim})"
        )
    if state.executed_actions:
        console.print("\nActions:")
        for executed in state.executed_actions:
            icon = _STATUS_ICONS.get(executed.status, "?")
            console.print(f"  {icon} {executed.action.describe()} — {executed.detail}")
    if state.pending_approval:
        console.print("\n[yellow]Held for human approval:[/yellow]")
        for held in state.pending_approval:
            console.print(f"  ⚠ {held.describe()} — {held.reason}")
    if state.escalated_to_human:
        console.print(f"\n[yellow]Escalated:[/yellow] {state.escalation_reason}")
    if state.tool_errors:
        console.print(f"[red]Tool errors:[/red] {state.tool_errors}")
    if state.error:
        console.print(f"[red]Run failed:[/red] {state.error}")


async def _run_async(repo: str, issue: int, dry_run: bool | None, yes: bool) -> int:
    from langgraph.checkpoint.memory import MemorySaver

    from triage_agent.agent.deps import TriageDeps
    from triage_agent.agent.llm import build_llm
    from triage_agent.github.client import GitHubClient
    from triage_agent.persistence.repositories import SqliteAuditStore
    from triage_agent.runner import TriageRunner

    settings = Settings()
    if dry_run is not None:
        settings.dry_run = dry_run
    configure_logging(settings.log_format, settings.log_level)
    config = load_app_config(settings.triage_config)
    owner, name = _parse_repo(repo)

    if not settings.github_token.get_secret_value():
        console.print("[red]GITHUB_TOKEN is not set[/red]")
        return 2

    interactive = sys.stdin.isatty() or yes
    client = GitHubClient(
        token=settings.github_token.get_secret_value(),
        security=config.security,
        mutations_enabled=not settings.dry_run,
    )
    store = await SqliteAuditStore.open(settings.triage_db_path)
    deps = TriageDeps(
        github=client,
        config=config,
        llm=build_llm(settings, config.llm),
        dry_run=settings.dry_run,
        approval_mode="interrupt" if interactive else "none",
    )
    runner = TriageRunner(deps, store, checkpointer=MemorySaver())

    mode = "[cyan]DRY RUN[/cyan]" if settings.dry_run else "[red]LIVE[/red]"
    console.print(f"Analyzing [bold]{repo}#{issue}[/bold]… ({mode})")
    try:
        outcome = await runner.run(owner, name, issue)
        if outcome.awaiting_approval:
            console.print(Panel(outcome.approval_request, title="Approval required"))
            if yes:
                console.print("[yellow]--yes given: approving all held actions[/yellow]")
                decisions: Any = True
            elif sys.stdin.isatty():
                decisions = [
                    {"action_index": i, "approved": typer.confirm(f"Approve [{i}] {a.describe()}?")}
                    for i, a in enumerate(outcome.state.pending_approval)
                ]
            else:
                console.print("[yellow]Non-interactive: held actions left unexecuted[/yellow]")
                decisions = False
            outcome = await runner.resume(outcome.run_id, decisions)
        _print_state(outcome.state)
        console.print(f"\nRun ID: {outcome.run_id}")
        return 1 if outcome.state.error else 0
    finally:
        await store.aclose()
        await client.aclose()


@app.command()
def run(
    repo: str = typer.Option(..., "--repo", help="Repository as owner/name"),
    issue: int = typer.Option(..., "--issue", help="Issue or PR number"),
    dry_run: bool | None = typer.Option(
        None,
        "--dry-run/--live",
        help="Override DRY_RUN from the environment",
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Approve all held actions without prompting (careful!)"
    ),
) -> None:
    """Triage one issue or pull request."""
    raise typer.Exit(asyncio.run(_run_async(repo, issue, dry_run, yes)))


@app.command()
def runs(limit: int = typer.Option(20, help="How many runs to list")) -> None:
    """List recent triage runs from the audit trail."""

    async def _list() -> None:
        from triage_agent.persistence.repositories import SqliteAuditStore

        settings = Settings()
        store = await SqliteAuditStore.open(settings.triage_db_path)
        try:
            for row in await store.list_runs(limit):
                console.print(
                    f"{row['id']}  {row['repository']}#{row['object_number']}  "
                    f"{row['final_status']}  {row['started_at']}"
                )
        finally:
            await store.aclose()

    asyncio.run(_list())


@app.command("show-run")
def show_run(run_id: str) -> None:
    """Show the full audit report for one run."""

    async def _show() -> None:
        import json

        from triage_agent.persistence.repositories import SqliteAuditStore

        settings = Settings()
        store = await SqliteAuditStore.open(settings.triage_db_path)
        try:
            report = await store.get_run_report(run_id)
            if report is None:
                console.print(f"[red]run {run_id} not found[/red]")
                raise typer.Exit(1)
            console.print_json(json.dumps(report, default=str))
        finally:
            await store.aclose()

    asyncio.run(_show())


@app.command("serve-mcp")
def serve_mcp() -> None:
    """Start the MCP server on stdio (for Claude Desktop etc.)."""
    from triage_agent.mcp.server import main as mcp_main

    mcp_main()


@app.command()
def eval(
    scenarios: str | None = typer.Option(None, help="Scenario directory override"),
    json_out: str | None = typer.Option(None, "--json", help="Write raw results JSON here"),
    live: bool = typer.Option(
        False, "--live", help="Use the real configured LLM instead of scripted outputs"
    ),
) -> None:
    """Run the evaluation suite and print the report."""
    from pathlib import Path

    from triage_agent.evals.metrics import render_report
    from triage_agent.evals.runner import run_evaluation

    async def _eval() -> int:
        metrics, results = await run_evaluation(
            Path(scenarios) if scenarios else None,
            json_output=Path(json_out) if json_out else None,
            live=live,
        )
        if live:
            console.print("[bold]LIVE MODEL evaluation[/bold]")
        console.print(render_report(metrics, results))
        return 0 if metrics.passed == metrics.scenarios else 1

    raise typer.Exit(asyncio.run(_eval()))


if __name__ == "__main__":
    app()
