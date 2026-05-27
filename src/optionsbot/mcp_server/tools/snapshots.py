"""latest_snapshot + score_breakdown (IBK-56, IBK-57)."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from sqlalchemy import desc, select

from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.serialization import iso_utc
from optionsbot.storage.schema import snapshots, strategy_scores


def register(server: FastMCP) -> None:
    """Attach the two snapshot read tools to the FastMCP server."""

    @server.tool()
    async def latest_snapshot(
        symbol: str,
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Return the latest persisted snapshot for a symbol with ALL scores."""
        symbol = symbol.upper().strip()
        lifespan = ctx.request_context.lifespan_context
        with lifespan.engine.connect() as conn:
            snap = conn.execute(
                select(snapshots)
                .where(snapshots.c.symbol == symbol)
                .order_by(desc(snapshots.c.ts))
                .limit(1)
            ).first()
            if snap is None:
                return {
                    "ok": False,
                    "error": "not_found",
                    "message": f"no snapshot for {symbol}",
                    "symbol": symbol,
                }
            score_rows = conn.execute(
                select(strategy_scores)
                .where(strategy_scores.c.snapshot_id == snap.id)
                .order_by(desc(strategy_scores.c.score))
            ).fetchall()
        return {
            "ok": True,
            "snapshot": {
                "id": snap.id,
                "symbol": snap.symbol,
                "ts": iso_utc(snap.ts),
                "spot": snap.spot,
                "iv_rank": snap.iv_rank,
                "hv20": snap.hv20,
                "iv_hv_ratio": snap.iv_hv_ratio,
                "expected_move": snap.expected_move,
                "regime_dir": snap.regime_dir,
                "regime_iv": snap.regime_iv,
                "raw_json": snap.raw_json,
            },
            "strategies": [
                {
                    "strategy": r.strategy,
                    "score": r.score,
                    "rationale": r.rationale,
                    "legs": r.legs_json or [],
                }
                for r in score_rows
            ],
        }

    @server.tool()
    async def score_breakdown(
        symbol: str,
        strategy: str,
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Return the score, rationale, and legs for a specific strategy on a symbol.

        Always reads from the latest snapshot per symbol -- "iron_condor on
        AAPL" returns the current scoring picture, not an arbitrary historical
        one.
        """
        symbol = symbol.upper().strip()
        lifespan = ctx.request_context.lifespan_context
        with lifespan.engine.connect() as conn:
            snap = conn.execute(
                select(snapshots.c.id, snapshots.c.ts)
                .where(snapshots.c.symbol == symbol)
                .order_by(desc(snapshots.c.ts))
                .limit(1)
            ).first()
            if snap is None:
                return {
                    "ok": False,
                    "error": "not_found",
                    "message": f"no snapshot for {symbol}",
                    "symbol": symbol,
                }
            score = conn.execute(
                select(strategy_scores)
                .where(strategy_scores.c.snapshot_id == snap.id)
                .where(strategy_scores.c.strategy == strategy)
                .limit(1)
            ).first()
        if score is None:
            return {
                "ok": False,
                "error": "not_found",
                "message": f"no score for strategy {strategy} on {symbol}",
                "symbol": symbol,
                "strategy": strategy,
            }
        return {
            "ok": True,
            "symbol": symbol,
            "strategy": score.strategy,
            "snapshot_id": snap.id,
            "snapshot_ts": iso_utc(snap.ts),
            "score": score.score,
            "rationale": score.rationale,
            "legs": score.legs_json or [],
        }
