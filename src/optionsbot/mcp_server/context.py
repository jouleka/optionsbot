"""Lifespan-scoped container for shared MCP server state.

Holds settings, the SQLAlchemy engine, and a lazily-instantiated
IBKRClient. Tools receive this via FastMCP's ``Context`` injection
(see ``ctx.request_context.lifespan_context`` in tool handlers).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy import Engine

from optionsbot.config import Settings, get_settings
from optionsbot.ibkr import IBKRClient
from optionsbot.storage.db import create_engine_for_path

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


@dataclass
class ServerContext:
    """Shared state available to every MCP tool via ctx.request_context.lifespan_context."""

    settings: Settings
    engine: Engine
    _ibkr: IBKRClient | None = field(default=None, repr=False)
    _ibkr_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def ibkr(self) -> IBKRClient:
        """Return the shared IBKRClient, constructing it on first call.

        The returned client is **not** auto-connected. This matches the
        rest of the IBKR layer pattern: every sub-client (ContractResolver,
        ChainClient, MarketDataClient, PositionsClient, HistoryClient) calls
        ``await client.ensure_connected()`` itself before issuing any IBKR
        call. Construct the sub-client of interest from this raw client and
        let it handle connection lifecycle.
        """
        async with self._ibkr_lock:
            if self._ibkr is None:
                self._ibkr = IBKRClient(role="mcp", settings=self.settings)
            return self._ibkr

    async def shutdown(self) -> None:
        """Close IBKR connection and dispose the SQLAlchemy engine."""
        if self._ibkr is not None:
            await self._ibkr.disconnect()
        self.engine.dispose()


@asynccontextmanager
async def app_lifespan(_server: FastMCP) -> AsyncIterator[ServerContext]:
    """FastMCP lifespan: build a ServerContext at startup, tear it down on exit."""
    settings = get_settings()
    engine = create_engine_for_path(settings.storage.db_path)
    ctx = ServerContext(settings=settings, engine=engine)
    try:
        yield ctx
    finally:
        await ctx.shutdown()
