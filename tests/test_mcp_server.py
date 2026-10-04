from __future__ import annotations

from typing import Any, cast

import pytest

from fakes import FakeGitHubClient, make_issue
from triage_agent.mcp.server import build_server
from triage_agent.mcp.tools import TriageToolService

REQUIRED_TOOLS = {
    "get_issue",
    "search_issues",
    "add_labels",
    "post_comment",
    "assign_issue",
    "close_issue",
    "get_repository_context",
    "get_pull_request",
}
EXTRA_TOOLS = {"reopen_issue", "get_recent_repository_activity"}


@pytest.fixture
def server_and_fake() -> tuple[Any, FakeGitHubClient]:
    fake = FakeGitHubClient()
    service = TriageToolService(cast(Any, fake), dry_run=True)
    return build_server(service), fake


async def test_all_required_tools_registered(server_and_fake: tuple[Any, FakeGitHubClient]) -> None:
    server, _ = server_and_fake
    tools = await server.list_tools()
    names = {tool.name for tool in tools}
    assert names >= REQUIRED_TOOLS
    assert names == REQUIRED_TOOLS | EXTRA_TOOLS


async def test_tools_have_descriptions_and_schemas(
    server_and_fake: tuple[Any, FakeGitHubClient],
) -> None:
    server, _ = server_and_fake
    for tool in await server.list_tools():
        assert tool.description, f"{tool.name} has no description"
        assert tool.input_schema.get("properties"), f"{tool.name} has no input schema"


async def test_call_tool_roundtrip(server_and_fake: tuple[Any, FakeGitHubClient]) -> None:
    server, fake = server_and_fake
    fake.issues[42] = make_issue(42)
    result = await server.call_tool(
        "get_issue", {"owner": "acme", "repo": "backend", "issue_number": 42}
    )
    assert result.structured_content is not None
    assert result.structured_content["number"] == 42


async def test_call_close_issue_without_confirm_is_safe(
    server_and_fake: tuple[Any, FakeGitHubClient],
) -> None:
    server, fake = server_and_fake
    result = await server.call_tool(
        "close_issue",
        {"owner": "acme", "repo": "backend", "issue_number": 42, "reason": "dup"},
    )
    assert result.structured_content is not None
    assert result.structured_content["status"] == "confirmation_required"
    assert fake.mutation_calls() == []
