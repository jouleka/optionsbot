"""MCP tool modules. Each module exposes register(server).

Keep this package initializer import-free: the restricted server must be able
to select safe modules without eagerly importing the broker-backed watchlist
implementation.
"""

__all__ = [
    "analyze",
    "daily_brief",
    "nightwatch",
    "positions",
    "snapshots",
    "track_record",
    "watchlist",
]
