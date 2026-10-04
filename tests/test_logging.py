from __future__ import annotations

from triage_agent.observability.logging import redact_secrets


def test_secret_keys_redacted() -> None:
    event = {
        "event": "call",
        "github_token": "ghp_abcdefghijklmnopqrstuvwxyz012345",
        "Authorization": "Bearer xyz",
        "detail": {"api_key": "sk-supersecretvalue1234567890"},
    }
    out = redact_secrets(None, "info", event)
    assert out["github_token"] == "[REDACTED]"
    assert out["Authorization"] == "[REDACTED]"
    assert out["detail"]["api_key"] == "[REDACTED]"


def test_secret_values_scrubbed_from_free_text() -> None:
    event = {"event": "oops leaked ghp_abcdefghijklmnopqrstuvwxyz012345 in a message"}
    out = redact_secrets(None, "info", event)
    assert "ghp_" not in out["event"]
    assert "[REDACTED]" in out["event"]


def test_normal_values_untouched() -> None:
    event = {"event": "triage", "repository": "acme/backend", "issue_number": 42}
    assert redact_secrets(None, "info", event) == event
