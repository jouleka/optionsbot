"""FastMCP factories for isolated FRED and Finnhub context servers."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from optionsbot.market_context.clients import FinnhubClient, FredClient


def build_fred_server(client: FredClient | None = None) -> FastMCP:
    """Build the read-only, allowlisted FRED MCP server."""
    fred = client or FredClient(os.environ.get("FRED_API_KEY", ""))
    server = FastMCP("optionsbot-fred-context")

    @server.tool()
    async def fred_macro_snapshot() -> dict[str, object]:
        """Return a fixed authoritative macro snapshot from approved FRED series."""
        return await fred.macro_snapshot()

    @server.tool()
    async def fred_series(series_id: str, limit: int = 12) -> dict[str, object]:
        """Return observations for one approved FRED series (maximum 120)."""
        return await fred.series(series_id, limit=limit)

    return server


def build_finnhub_server(client: FinnhubClient | None = None) -> FastMCP:
    """Build the read-only Finnhub free-tier MCP server."""
    finnhub = client or FinnhubClient(os.environ.get("FINNHUB_API_KEY", ""))
    server = FastMCP("optionsbot-finnhub-context")

    @server.tool()
    async def finnhub_quote(symbol: str) -> dict[str, object]:
        """Return a bounded real-time US equity quote for corroboration context."""
        return await finnhub.quote(symbol)

    @server.tool()
    async def finnhub_company_news(
        symbol: str, days: int = 3, limit: int = 10
    ) -> dict[str, object]:
        """Return capped, sanitized company news; all prose is explicitly untrusted."""
        return await finnhub.company_news(symbol, days=days, limit=limit)

    @server.tool()
    async def finnhub_earnings_calendar(
        symbol: str = "", days: int = 14, limit: int = 50
    ) -> dict[str, object]:
        """Return the bounded US earnings calendar; dates require independent confirmation."""
        return await finnhub.earnings_calendar(
            symbol=symbol or None,
            days=days,
            limit=limit,
        )

    return server
