"""MCP server exposing the GitHub triage tools.

Run over stdio (for Claude Desktop and other MCP clients):

    python -m triage_agent.mcp.server

The eight required tools:
    get_issue, search_issues, add_labels, post_comment, assign_issue,
    close_issue (destructive — human confirmation required),
    get_repository_context, get_pull_request
plus two extras: reopen_issue (also gated) and get_recent_repository_activity.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.mcpserver import MCPServer

from triage_agent.config import Settings, load_app_config
from triage_agent.github.client import GitHubClient
from triage_agent.mcp.tools import TriageToolService
from triage_agent.observability.logging import configure_logging


def build_server(service: TriageToolService) -> MCPServer:
    """Wire the tool service into an MCP server instance."""
    server = MCPServer(
        name="github-triage",
        instructions=(
            "GitHub issue and pull-request triage tools. Read tools are always "
            "safe. Mutating tools respect dry-run mode. close_issue and "
            "reopen_issue are destructive: they require confirm=true, which "
            "must only be sent after a human has explicitly approved the "
            "proposed action shown by the first call."
        ),
        version="0.1.0",
    )

    @server.tool()
    async def get_issue(owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        """Fetch full issue metadata including labels, assignees and comments."""
        return await service.get_issue(owner, repo, issue_number)

    @server.tool()
    async def search_issues(owner: str, repo: str, query: str) -> dict[str, Any]:
        """Search issues/PRs in one repository (GitHub search syntax) to find
        related or duplicate reports."""
        return await service.search_issues(owner, repo, query)

    @server.tool()
    async def add_labels(
        owner: str, repo: str, issue_number: int, labels: list[str]
    ) -> dict[str, Any]:
        """Add one or more labels to an issue or pull request."""
        return await service.add_labels(owner, repo, issue_number, labels)

    @server.tool()
    async def post_comment(
        owner: str, repo: str, issue_number: int, comment: str
    ) -> dict[str, Any]:
        """Post a comment on an issue or pull request."""
        return await service.post_comment(owner, repo, issue_number, comment)

    @server.tool()
    async def assign_issue(
        owner: str, repo: str, issue_number: int, assignees: list[str]
    ) -> dict[str, Any]:
        """Assign one or more GitHub users. Users are verified as assignable
        first; unverifiable users are rejected, never guessed."""
        return await service.assign_issue(owner, repo, issue_number, assignees)

    @server.tool()
    async def close_issue(
        owner: str,
        repo: str,
        issue_number: int,
        reason: str,
        state_reason: str = "not_planned",
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Close an issue. DESTRUCTIVE: the first call returns a proposed
        action for human review and does NOT close anything. Only call again
        with confirm=true after the human has explicitly approved."""
        return await service.close_issue(
            owner, repo, issue_number, reason, state_reason, confirm
        )

    @server.tool()
    async def get_repository_context(owner: str, repo: str) -> dict[str, Any]:
        """Fetch repository decision context: description, topics, README,
        CONTRIBUTING.md, CODEOWNERS and the available labels."""
        return await service.get_repository_context(owner, repo)

    @server.tool()
    async def get_pull_request(owner: str, repo: str, pr_number: int) -> dict[str, Any]:
        """Fetch a pull request with changed files, commits, reviews, CI check
        status and linked issues."""
        return await service.get_pull_request(owner, repo, pr_number)

    @server.tool()
    async def reopen_issue(
        owner: str, repo: str, issue_number: int, reason: str, confirm: bool = False
    ) -> dict[str, Any]:
        """Reopen a closed issue. DESTRUCTIVE: requires confirm=true after
        explicit human approval, same flow as close_issue."""
        return await service.reopen_issue(owner, repo, issue_number, reason, confirm)

    @server.tool()
    async def get_recent_repository_activity(
        owner: str, repo: str, days: int = 14
    ) -> dict[str, Any]:
        """List recently updated issues/PRs (default: last 14 days)."""
        return await service.get_recent_repository_activity(owner, repo, days)

    return server


def create_default_server() -> tuple[MCPServer, GitHubClient]:
    """Build the server from environment settings (production entrypoint)."""
    settings = Settings()
    configure_logging(settings.log_format, settings.log_level)
    app_config = load_app_config(settings.triage_config)
    client = GitHubClient(
        token=settings.github_token.get_secret_value(),
        security=app_config.security,
        # Defence in depth: in dry-run the client itself refuses mutations.
        mutations_enabled=not settings.dry_run,
    )
    service = TriageToolService(client, dry_run=settings.dry_run)
    return build_server(service), client


async def amain() -> None:
    server, client = create_default_server()
    try:
        await server.run_stdio_async()
    finally:
        await client.aclose()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
