"""positions tool (IBK-112): live open-book read surface.

Returns the account's open positions grouped by underlying, each leg enriched with
live P&L (from ib.portfolio()), DTE, and best-effort Greeks.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from optionsbot.analysis.positions import assemble_open_book
from optionsbot.ibkr.market_data import MarketDataClient
from optionsbot.ibkr.positions import PositionsClient
from optionsbot.mcp_server.context import ServerContext

log = logging.getLogger(__name__)


def register(server: FastMCP) -> None:
    @server.tool()
    async def positions(
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Live open book: positions grouped by underlying with P&L, DTE, and Greeks.

        Returns ``{ok, as_of, net_unrealized_pnl, group_count, position_count,
        groups:[{underlying, net_unrealized_pnl, legs:[...]}]}`` on success, or
        ``{ok: false, error: "ibkr_unavailable"}`` when IBKR can't be reached.
        """
        lifespan = ctx.request_context.lifespan_context
        client = await lifespan.ibkr()
        pos_client = PositionsClient(client)
        md_client = MarketDataClient(client)
        try:
            view = await assemble_open_book(pos_client, md_client, datetime.now(UTC))
        except Exception:  # noqa: BLE001 -- read tool: structured failure, never 500
            log.exception("positions tool failed to read the open book")
            return {
                "ok": False,
                "error": "ibkr_unavailable",
                "message": "could not reach IBKR to read positions",
            }
        return {"ok": True, **view}
