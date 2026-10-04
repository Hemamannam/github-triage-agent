from __future__ import annotations

import pytest

from triage_agent.config import SecurityConfig
from triage_agent.github.client import GitHubClient


@pytest.fixture
def gh_client() -> GitHubClient:
    # Tiny backoff so retry paths execute without slowing the suite down.
    return GitHubClient(token="test-token", max_attempts=3, backoff_base=0.01, backoff_cap=0.05)


@pytest.fixture
def readonly_client() -> GitHubClient:
    return GitHubClient(token="test-token", mutations_enabled=False)


@pytest.fixture
def allowlisted_client() -> GitHubClient:
    return GitHubClient(
        token="test-token",
        security=SecurityConfig(allowed_repositories=["acme/backend"]),
    )


def issue_payload(number: int = 1, **overrides: object) -> dict:
    payload: dict = {
        "number": number,
        "title": "Login crashes with expired token",
        "body": "Steps to reproduce: ...",
        "state": "open",
        "user": {"login": "alice"},
        "labels": [{"name": "bug", "color": "d73a4a"}],
        "assignees": [],
        "comments": 0,
        "author_association": "NONE",
        "html_url": f"https://github.com/acme/backend/issues/{number}",
    }
    payload.update(overrides)
    return payload
