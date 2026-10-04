"""Run orchestration: graph execution + audit trail + run lifecycle.

The runner owns everything around a triage run that isn't graph logic:
binding the logging context, persisting the audit trail, detecting the
approval interrupt, and recording human decisions on resume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from triage_agent.agent.deps import TriageDeps
from triage_agent.agent.graph import build_graph
from triage_agent.agent.state import ProposedAction, TriageState
from triage_agent.observability.logging import bind_run_context, clear_run_context
from triage_agent.persistence.repositories import AuditStore, NullAuditStore
from triage_agent.safety.approvals import parse_resume_payload

logger = structlog.get_logger("runner")


@dataclass
class RunOutcome:
    run_id: str
    state: TriageState
    awaiting_approval: bool = False
    approval_request: str = ""


class TriageRunner:
    def __init__(
        self,
        deps: TriageDeps,
        store: AuditStore | None = None,
        *,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> None:
        self._deps = deps
        self._store: AuditStore = store if store is not None else NullAuditStore()
        self._graph = build_graph(deps, checkpointer=checkpointer)

    async def run(self, owner: str, repo: str, issue_number: int) -> RunOutcome:
        initial = TriageState(owner=owner, repo=repo, issue_number=issue_number)
        run_id = initial.run_id
        bind_run_context(
            run_id=run_id, repository=f"{owner}/{repo}", issue_number=issue_number
        )
        logger.info("run.started", dry_run=self._deps.dry_run)
        await self._store.start_run(run_id, f"{owner}/{repo}", issue_number, "unknown")
        try:
            raw = await self._graph.ainvoke(
                initial, {"configurable": {"thread_id": run_id}}
            )
            return await self._finalize(run_id, raw)
        finally:
            clear_run_context()

    async def resume(self, run_id: str, resume_value: Any) -> RunOutcome:
        """Resume an interrupted run with the human's approval decisions."""
        config: RunnableConfig = {"configurable": {"thread_id": run_id}}
        bind_run_context(run_id=run_id)
        try:
            snapshot = await self._graph.aget_state(config)
            pending_raw = (snapshot.values or {}).get("pending_approval", [])
            pending = [ProposedAction.model_validate(p) for p in pending_raw]

            raw = await self._graph.ainvoke(Command(resume=resume_value), config)

            decisions = parse_resume_payload(resume_value, pending)
            decided = {d.action_index: d for d in decisions}
            for i, actn in enumerate(pending):
                decision = decided.get(i)
                await self._store.record_approval(
                    run_id,
                    actn,
                    bool(decision and decision.approved),
                    note=decision.note if decision else "no explicit decision",
                )
            return await self._finalize(run_id, raw)
        finally:
            clear_run_context()

    async def _finalize(self, run_id: str, raw: dict[str, Any]) -> RunOutcome:
        if "__interrupt__" in raw:
            payload = raw["__interrupt__"][0].value
            state = TriageState.model_validate(
                {k: v for k, v in raw.items() if k != "__interrupt__"}
            )
            request = str(payload.get("request", ""))
            await self._store.complete_run(run_id, "awaiting_approval", request)
            logger.info("run.awaiting_approval", held=len(state.pending_approval))
            return RunOutcome(
                run_id=run_id,
                state=state,
                awaiting_approval=True,
                approval_request=request,
            )

        state = TriageState.model_validate(raw)
        await self._store.record_decision(run_id, state)
        for executed in state.executed_actions:
            await self._store.record_tool_execution(run_id, executed)
        final_status = "failed" if state.error else "completed"
        await self._store.complete_run(run_id, final_status, state.final_summary)
        logger.info(
            "run.finished",
            status=final_status,
            executed=len(state.executed_actions),
            escalated=state.escalated_to_human,
        )
        return RunOutcome(run_id=run_id, state=state)
