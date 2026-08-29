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
from sqlalchemy import select

from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.serialization import iso_utc
from optionsbot.storage.schema import orders, position_settlements

log = logging.getLogger(__name__)


async def assemble_open_book(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy compatibility seam; keeps IBKR modules out of restricted imports."""
    from optionsbot.analysis.positions import assemble_open_book as implementation

    return await implementation(*args, **kwargs)


def register(server: FastMCP) -> None:
    @server.tool()
    async def positions(
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Live open book: positions grouped by underlying with P&L, DTE, and Greeks.

        Returns ``{ok, as_of, net_unrealized_pnl, group_count, position_count,
        groups:[{underlying, net_unrealized_pnl, legs:[...]}], beta_weighted}`` on success,
        or ``{ok: false, error: "ibkr_unavailable"}`` when IBKR can't be reached.
        ``beta_weighted`` (IBK-118) is the SPY-comparable book delta (None if the benchmark
        history is unavailable), or absent when portfolio beta-weighting is disabled.
        """
        lifespan = ctx.request_context.lifespan_context
        if not getattr(lifespan, "broker_access", True):
            return _persisted_open_book(lifespan)

        from optionsbot.ibkr.history import HistoryClient
        from optionsbot.ibkr.market_data import MarketDataClient
        from optionsbot.ibkr.positions import PositionsClient

        client = await lifespan.ibkr()
        settings = lifespan.settings
        pos_client = PositionsClient(client)
        md_client = MarketDataClient(client)
        history_client = HistoryClient(client) if settings.portfolio.enabled else None
        try:
            view = await assemble_open_book(
                pos_client,
                md_client,
                datetime.now(UTC),
                history_client=history_client,
                benchmark_symbol=settings.scan.benchmark_symbol,
                beta_window=settings.portfolio.beta_window,
            )
        except Exception:  # noqa: BLE001 -- read tool: structured failure, never 500
            log.exception("positions tool failed to read the open book")
            return {
                "ok": False,
                "error": "ibkr_unavailable",
                "message": "could not reach IBKR to read positions",
            }
        return {"ok": True, **view}


def _persisted_open_book(lifespan: ServerContext) -> dict[str, Any]:
    """Ledger-only fallback for the restricted endpoint (never contacts IBKR)."""
    with lifespan.engine.connect() as conn:
        entries = conn.execute(
            select(orders)
            .where(orders.c.intent == "open")
            .where(orders.c.status == "filled")
            .order_by(orders.c.id)
        ).fetchall()
        closes = conn.execute(
            select(orders.c.closes_order_id)
            .where(orders.c.intent == "close")
            .where(orders.c.status == "filled")
            .where(orders.c.closes_order_id.is_not(None))
        ).fetchall()
        settlements = conn.execute(
            select(position_settlements.c.entry_order_id)
        ).fetchall()
    closed_ids = {int(row.closes_order_id) for row in closes}
    settled_ids = {int(row.entry_order_id) for row in settlements}
    terminal_ids = closed_ids | settled_ids
    open_rows = [row for row in entries if int(row.id) not in terminal_ids]
    return {
        "ok": True,
        "as_of": datetime.now(UTC).isoformat(),
        "source": "persisted_order_ledger",
        "live_quotes_available": False,
        "position_count": len(open_rows),
        "positions": [
            {
                "id": int(row.id),
                "symbol": row.symbol,
                "strategy": row.strategy,
                "quantity": row.quantity,
                "legs": list(row.legs_json or []),
                "filled_at": iso_utc(row.terminal_ts),
            }
            for row in open_rows
        ],
        "note": "restricted MCP has no broker access; live P&L/Greeks are unavailable",
    }
