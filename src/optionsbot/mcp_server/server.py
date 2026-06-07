"""FastMCP server factory."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from optionsbot.mcp_server.context import app_lifespan


def build_server() -> FastMCP:
    """Build the optionsbot MCP server with lifespan + all tools registered."""
    server = FastMCP("optionsbot", lifespan=app_lifespan)
    _register_tools(server)
    return server


def _register_tools(server: FastMCP) -> None:
    from optionsbot.mcp_server.tools import analyze, daily_brief, snapshots, watchlist

    watchlist.register(server)
    analyze.register(server)
    snapshots.register(server)
    daily_brief.register(server)
