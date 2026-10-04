"""Database layer for the audit trail.

SQLite today; the ``AuditStore`` protocol in ``repositories.py`` is the only
surface the rest of the application sees, so moving to PostgreSQL means
writing one new implementation — no call sites change.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
    id            TEXT PRIMARY KEY,
    repository    TEXT NOT NULL,
    object_number INTEGER NOT NULL,
    object_type   TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    completed_at  TEXT,
    final_status  TEXT NOT NULL DEFAULT 'running',
    summary       TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS agent_decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL REFERENCES agent_runs(id),
    classification    TEXT,
    confidence        REAL,
    severity          TEXT,
    priority          TEXT,
    reasoning_summary TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_executions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES agent_runs(id),
    tool        TEXT NOT NULL,
    input       TEXT NOT NULL,
    result      TEXT NOT NULL,
    success     INTEGER NOT NULL,
    latency_ms  REAL NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS human_approvals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES agent_runs(id),
    proposed_action TEXT NOT NULL,
    approved        INTEGER NOT NULL,
    approved_at     TEXT NOT NULL,
    note            TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_decisions_run ON agent_decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_executions_run ON tool_executions(run_id);
CREATE INDEX IF NOT EXISTS idx_approvals_run ON human_approvals(run_id);
"""


async def open_database(path: Path | str) -> aiosqlite.Connection:
    """Open (and initialise) the audit database."""
    conn = await aiosqlite.connect(str(path))
    conn.row_factory = aiosqlite.Row
    await conn.executescript(SCHEMA)
    await conn.commit()
    return conn
