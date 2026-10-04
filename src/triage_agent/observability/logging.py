"""Structured JSON logging via structlog.

Every agent execution binds a ``run_id`` so all lines from one triage run can
be correlated. A redaction processor scrubs anything that looks like a secret
before it can reach a log sink — tokens must never appear in logs.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog

_SECRET_KEY_PATTERN = re.compile(
    r"(token|secret|password|api_key|apikey|authorization)", re.IGNORECASE
)
_SECRET_VALUE_PATTERN = re.compile(
    r"(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9\-_]{20,})"
)


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _SECRET_VALUE_PATTERN.sub("[REDACTED]", value)
    if isinstance(value, dict):
        return {
            k: "[REDACTED]" if _SECRET_KEY_PATTERN.search(str(k)) else _redact(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    return value


def redact_secrets(
    _logger: Any, _method_name: str, event_dict: structlog.typing.EventDict
) -> structlog.typing.EventDict:
    """structlog processor: scrub secret-looking keys and values."""
    return {k: _redact(v) if not _SECRET_KEY_PATTERN.search(k) else "[REDACTED]"
            for k, v in event_dict.items()}


def configure_logging(log_format: str = "json", level: str = "INFO") -> None:
    """Configure structlog once at process start."""
    renderer: structlog.typing.Processor
    if log_format == "console":
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_secrets,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        cache_logger_on_first_use=True,
    )


def bind_run_context(**kwargs: Any) -> None:
    """Bind run-scoped fields (run_id, repository, issue_number, ...)."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_run_context() -> None:
    structlog.contextvars.clear_contextvars()
