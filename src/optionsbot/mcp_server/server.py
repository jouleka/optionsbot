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
    """Hook for tool modules to register their @server.tool() handlers.

    Filled in by Task 2 (watchlist), Task 3 (analyze), Task 4 (snapshot tools).
    """
    from optionsbot.mcp_server.tools import watchlist

    watchlist.register(server)
