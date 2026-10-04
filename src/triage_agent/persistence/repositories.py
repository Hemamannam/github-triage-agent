"""Audit-trail store: every run, decision, tool call and approval persists."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import aiosqlite

from triage_agent.agent.state import ExecutedAction, ProposedAction, TriageState
from triage_agent.persistence.database import open_database


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AuditStore(Protocol):
    """Persistence surface used by the runner. Swap implementations freely."""

    async def start_run(
        self, run_id: str, repository: str, object_number: int, object_type: str
    ) -> None: ...

    async def complete_run(self, run_id: str, final_status: str, summary: str) -> None: ...

    async def record_decision(self, run_id: str, state: TriageState) -> None: ...

    async def record_tool_execution(self, run_id: str, executed: ExecutedAction) -> None: ...

    async def record_approval(
        self, run_id: str, action: ProposedAction, approved: bool, note: str = ""
    ) -> None: ...

    async def get_run_report(self, run_id: str) -> dict[str, Any] | None: ...

    async def list_runs(self, limit: int = 20) -> list[dict[str, Any]]: ...

    async def aclose(self) -> None: ...


class SqliteAuditStore:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def open(cls, path: Path | str) -> SqliteAuditStore:
        return cls(await open_database(path))

    async def aclose(self) -> None:
        await self._conn.close()

    # ------------------------------------------------------------------ writes

    async def start_run(
        self, run_id: str, repository: str, object_number: int, object_type: str
    ) -> None:
        await self._conn.execute(
            "INSERT OR REPLACE INTO agent_runs "
            "(id, repository, object_number, object_type, started_at, final_status) "
            "VALUES (?, ?, ?, ?, ?, 'running')",
            (run_id, repository, object_number, object_type, _now()),
        )
        await self._conn.commit()

    async def complete_run(self, run_id: str, final_status: str, summary: str) -> None:
        await self._conn.execute(
            "UPDATE agent_runs SET completed_at = ?, final_status = ?, summary = ? "
            "WHERE id = ?",
            (_now(), final_status, summary, run_id),
        )
        await self._conn.commit()

    async def record_decision(self, run_id: str, state: TriageState) -> None:
        assessment: Any = (
            state.pr_assessment
            if state.object_type.value == "pull_request"
            else state.issue_assessment
        )
        if assessment is None:
            return
        await self._conn.execute(
            "INSERT INTO agent_decisions "
            "(run_id, classification, confidence, severity, priority, "
            " reasoning_summary, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                assessment.classification.value,
                assessment.confidence,
                assessment.severity.value,
                assessment.priority.value,
                assessment.reasoning_summary,
                _now(),
            ),
        )
        await self._conn.commit()

    async def record_tool_execution(self, run_id: str, executed: ExecutedAction) -> None:
        await self._conn.execute(
            "INSERT INTO tool_executions "
            "(run_id, tool, input, result, success, latency_ms, retry_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                executed.action.kind.value,
                json.dumps(executed.action.params, default=str),
                f"{executed.status.value}: {executed.detail}",
                int(executed.status.value in ("ok", "dry_run", "skipped")),
                executed.latency_ms,
                executed.retries,
                _now(),
            ),
        )
        await self._conn.commit()

    async def record_approval(
        self, run_id: str, action: ProposedAction, approved: bool, note: str = ""
    ) -> None:
        await self._conn.execute(
            "INSERT INTO human_approvals "
            "(run_id, proposed_action, approved, approved_at, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, action.describe(), int(approved), _now(), note),
        )
        await self._conn.commit()

    # ------------------------------------------------------------------- reads

    async def get_run_report(self, run_id: str) -> dict[str, Any] | None:
        run = await self._fetchone("SELECT * FROM agent_runs WHERE id = ?", (run_id,))
        if run is None:
            return None
        return {
            "run": run,
            "decisions": await self._fetchall(
                "SELECT * FROM agent_decisions WHERE run_id = ? ORDER BY id", (run_id,)
            ),
            "tool_executions": await self._fetchall(
                "SELECT * FROM tool_executions WHERE run_id = ? ORDER BY id", (run_id,)
            ),
            "approvals": await self._fetchall(
                "SELECT * FROM human_approvals WHERE run_id = ? ORDER BY id", (run_id,)
            ),
        }

    async def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return await self._fetchall(
            "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT ?", (limit,)
        )

    async def _fetchone(self, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        async with self._conn.execute(sql, params) as cursor:
            row = await cursor.fetchone()
        return dict(row) if row is not None else None

    async def _fetchall(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        async with self._conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]


class NullAuditStore:
    """No-op store for contexts where persistence is unwanted (some tests)."""

    async def start_run(self, *args: Any, **kwargs: Any) -> None: ...
    async def complete_run(self, *args: Any, **kwargs: Any) -> None: ...
    async def record_decision(self, *args: Any, **kwargs: Any) -> None: ...
    async def record_tool_execution(self, *args: Any, **kwargs: Any) -> None: ...
    async def record_approval(self, *args: Any, **kwargs: Any) -> None: ...
    async def get_run_report(self, run_id: str) -> dict[str, Any] | None:
        return None
    async def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return []
    async def aclose(self) -> None: ...
