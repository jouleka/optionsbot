"""MCP tool modules. Each module exposes register(server)."""

from optionsbot.mcp_server.tools import (
    analyze,
    daily_brief,
    nightwatch,
    positions,
    snapshots,
    track_record,
    watchlist,
)

__all__ = [
    "analyze",
    "daily_brief",
    "nightwatch",
    "positions",
    "snapshots",
    "track_record",
    "watchlist",
]
