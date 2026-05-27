"""Watchlist tools (IBK-51..53, IBK-55)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.exc import IntegrityError

from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.serialization import iso_utc
from optionsbot.storage.schema import snapshots, watchlist

_VALID_DIRECTIONS = frozenset({"bull", "neutral", "bear"})
_VALID_IV_REGIMES = frozenset({"high", "neutral", "low"})


def register(server: FastMCP) -> None:
    """Attach the four watchlist tools to the FastMCP server."""

    @server.tool()
    async def add_to_watchlist(
        symbol: str,
        notes: str | None,
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Add a ticker to the watchlist after validating it via IBKR.

        Idempotent: re-adding an existing symbol succeeds with
        ``already_present=True`` instead of raising.
        """
        symbol = symbol.upper().strip()
        if not symbol:
            return {"ok": False, "error": "invalid_symbol", "message": "symbol is empty"}
        lifespan = ctx.request_context.lifespan_context
        try:
            ibkr = await lifespan.ibkr()
            resolver = ContractResolver(ibkr)
            await resolver.stock(symbol)
        except ConnectionError as e:
            return {
                "ok": False,
                "error": "ibkr_unavailable",
                "message": str(e),
                "symbol": symbol,
            }
        except ValueError as e:
            return {
                "ok": False,
                "error": "unknown_symbol",
                "message": str(e),
                "symbol": symbol,
            }
        added_at = datetime.now(UTC)
        try:
            with lifespan.engine.begin() as conn:
                conn.execute(
                    insert(watchlist).values(
                        symbol=symbol, notes=notes, added_at=added_at
                    )
                )
        except IntegrityError:
            return {"ok": True, "symbol": symbol, "already_present": True}
        return {"ok": True, "symbol": symbol, "added_at": added_at.isoformat()}

    @server.tool()
    async def remove_from_watchlist(
        symbol: str,
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Remove a ticker from the watchlist. Snapshot history is preserved."""
        symbol = symbol.upper().strip()
        lifespan = ctx.request_context.lifespan_context
        with lifespan.engine.begin() as conn:
            result = conn.execute(delete(watchlist).where(watchlist.c.symbol == symbol))
        return {"ok": True, "symbol": symbol, "removed": result.rowcount > 0}

    @server.tool()
    async def list_watchlist(
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Return all watchlist entries with view overrides and last-scanned ts."""
        lifespan = ctx.request_context.lifespan_context
        last_scanned_sq = (
            select(snapshots.c.symbol, func.max(snapshots.c.ts).label("last_scanned"))
            .group_by(snapshots.c.symbol)
            .subquery()
        )
        stmt = (
            select(
                watchlist.c.symbol,
                watchlist.c.view_override_dir,
                watchlist.c.view_override_iv,
                watchlist.c.notes,
                watchlist.c.added_at,
                last_scanned_sq.c.last_scanned,
            )
            .select_from(
                watchlist.outerjoin(
                    last_scanned_sq, watchlist.c.symbol == last_scanned_sq.c.symbol
                )
            )
            .order_by(watchlist.c.symbol)
        )
        with lifespan.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        entries = [
            {
                "symbol": r.symbol,
                "view_override": {
                    "direction": r.view_override_dir,
                    "iv_regime": r.view_override_iv,
                },
                "notes": r.notes,
                "added_at": iso_utc(r.added_at),
                "last_scanned": iso_utc(r.last_scanned),
            }
            for r in rows
        ]
        return {"ok": True, "entries": entries}

    @server.tool()
    async def set_view_override(
        symbol: str,
        direction: str | None,
        iv_regime: str | None,
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Pin a view override (or clear it by passing null for either field).

        Passing ``direction=None`` clears the direction override; same for
        ``iv_regime``. To pin both, pass both. To clear both, pass both as null.
        """
        symbol = symbol.upper().strip()
        if direction is not None and direction not in _VALID_DIRECTIONS:
            return {
                "ok": False,
                "error": "invalid_direction",
                "message": f"direction must be one of {sorted(_VALID_DIRECTIONS)} or null",
            }
        if iv_regime is not None and iv_regime not in _VALID_IV_REGIMES:
            return {
                "ok": False,
                "error": "invalid_iv_regime",
                "message": f"iv_regime must be one of {sorted(_VALID_IV_REGIMES)} or null",
            }
        lifespan = ctx.request_context.lifespan_context
        with lifespan.engine.begin() as conn:
            result = conn.execute(
                update(watchlist)
                .where(watchlist.c.symbol == symbol)
                .values(view_override_dir=direction, view_override_iv=iv_regime)
            )
        if result.rowcount == 0:
            return {
                "ok": False,
                "error": "unknown_symbol",
                "message": f"{symbol} is not in the watchlist",
            }
        return {
            "ok": True,
            "symbol": symbol,
            "override": {"direction": direction, "iv_regime": iv_regime},
        }
