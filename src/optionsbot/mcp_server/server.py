"""FastMCP server factory."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


def build_server(*, restricted: bool = False) -> FastMCP:
    """Build the optionsbot MCP server with lifespan + all tools registered."""
    if restricted:
        from optionsbot.mcp_server.restricted_context import restricted_app_lifespan

        server = FastMCP("optionsbot", lifespan=restricted_app_lifespan)
        _register_restricted_tools(server)
        return server

    from optionsbot.mcp_server.context import app_lifespan

    server = FastMCP("optionsbot", lifespan=app_lifespan)
    _register_tools(server)
    return server


def _register_tools(server: FastMCP) -> None:
    from optionsbot.mcp_server.tools import (
        analyze,
        daily_brief,
        nightwatch,
        positions,
        snapshots,
        track_record,
        watchlist,
    )

    watchlist.register(server)
    analyze.register(server)
    snapshots.register(server)
    daily_brief.register(server)
    positions.register(server)
    track_record.register(server)
    nightwatch.register(server)


def _register_restricted_tools(server: FastMCP) -> None:
    from optionsbot.mcp_server.tools import (
        nightwatch,
        positions,
        snapshots,
    )
    from optionsbot.mcp_server.tools.restricted import register as register_restricted

    snapshots.register(server)
    positions.register(server)
    nightwatch.register(server)
    register_restricted(server)
