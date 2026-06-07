"""analyze tool (IBK-54).

If ``fresh=True``, runs scan_symbol (live IBKR fetch + persist).
If ``fresh=False``, returns the latest persisted snapshot's view + top-K
strategies; returns a structured ``no_snapshot`` error when nothing exists.
"""

from __future__ import annotations

from typing import Any, cast

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from sqlalchemy import desc, select

from optionsbot.analysis.types import Direction, IVRegime
from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.serialization import dump_scored, dump_view, iso_utc
from optionsbot.scan import scan_symbol
from optionsbot.scoring import DEFAULT_THRESHOLD, DEFAULT_TOP_K, top_k
from optionsbot.scoring.composite import has_positive_edge
from optionsbot.storage.schema import snapshots, strategy_scores, watchlist


def register(server: FastMCP) -> None:
    @server.tool()
    async def analyze(
        symbol: str,
        fresh: bool,
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Analyze a symbol: live scan when fresh=true, cached snapshot when fresh=false.

        Returns ``{ok, symbol, snapshot_id, snapshot_ts, view, top_strategies}``
        on success. Returns ``{ok: false, error: "no_snapshot", hint: ...}``
        when fresh=false and no snapshot has been persisted yet.
        """
        symbol = symbol.upper().strip()
        lifespan = ctx.request_context.lifespan_context
        if fresh:
            return await _analyze_fresh(symbol, lifespan)
        return _analyze_cached(symbol, lifespan)


async def _analyze_fresh(symbol: str, lifespan: ServerContext) -> dict[str, Any]:
    override = _watchlist_view_override(symbol, lifespan)
    ibkr = await lifespan.ibkr()
    result = await scan_symbol(
        symbol,
        ibkr,
        lifespan.engine,
        lifespan.settings,
        view_override=override,
    )
    selected = top_k(
        result.scored, k=DEFAULT_TOP_K, threshold=DEFAULT_THRESHOLD,
        rank_by="expectancy",
    )
    return {
        "ok": True,
        "symbol": result.symbol,
        "snapshot_id": result.snapshot_id,
        "snapshot_ts": iso_utc(result.snapshot_ts),
        "view": dump_view(result.view),
        "no_positive_edge": not any(has_positive_edge(s.suggestion) for s in selected),
        "top_strategies": [dump_scored(s) for s in selected],
    }


def _analyze_cached(symbol: str, lifespan: ServerContext) -> dict[str, Any]:
    with lifespan.engine.connect() as conn:
        row = conn.execute(
            select(snapshots)
            .where(snapshots.c.symbol == symbol)
            .order_by(desc(snapshots.c.ts))
            .limit(1)
        ).first()
        if row is None:
            return {
                "ok": False,
                "error": "no_snapshot",
                "message": f"No persisted snapshot for {symbol}.",
                "hint": "retry with fresh=true to fetch live from IBKR",
                "symbol": symbol,
            }
        score_rows = conn.execute(
            select(strategy_scores)
            .where(strategy_scores.c.snapshot_id == row.id)
            .order_by(desc(strategy_scores.c.score))
        ).fetchall()
    # direction_strength and earnings_in_window are not persisted on the
    # snapshots table -- both are derivable on the next fresh scan. The fresh
    # path returns concrete values for both via dump_view(MarketView).
    view = {
        "direction": row.regime_dir,
        "direction_strength": None,
        "iv_regime": row.regime_iv,
        "iv_rank_value": row.iv_rank,
        "earnings_in_window": None,
        "warming_up": row.raw_json.get("warming_up") if row.raw_json else None,
    }
    # NOTE (IBK-104): this cached path stays SCORE-ranked. strategy_scores rows
    # don't persist expected_value/max_loss, so risk-normalized edge ranking
    # isn't computable here without a schema migration; the fresh path edge-ranks.
    # Mirrors top_k(scored, k=DEFAULT_TOP_K, threshold=DEFAULT_THRESHOLD) but
    # operates on DB rows instead of ScoredStrategy objects. The score_rows
    # query above already orders by `desc(score)`, so filtering then slicing
    # produces the same selection. If top_k ever grows tie-breaking or
    # diversity logic, update this branch to match.
    selected = [r for r in score_rows if r.score >= DEFAULT_THRESHOLD][:DEFAULT_TOP_K]
    return {
        "ok": True,
        "symbol": symbol,
        "snapshot_id": row.id,
        "snapshot_ts": iso_utc(row.ts),
        "view": view,
        "top_strategies": [
            {
                "strategy_name": r.strategy,
                "score": r.score,
                "rationale": r.rationale,
                "legs": r.legs_json or [],
            }
            for r in selected
        ],
    }


def _watchlist_view_override(
    symbol: str, lifespan: ServerContext
) -> tuple[Direction | None, IVRegime | None] | None:
    """Look up the watchlist row's view override columns. Returns None if no entry."""
    with lifespan.engine.connect() as conn:
        row = conn.execute(
            select(watchlist.c.view_override_dir, watchlist.c.view_override_iv).where(
                watchlist.c.symbol == symbol
            )
        ).first()
    if row is None:
        return None
    if row.view_override_dir is None and row.view_override_iv is None:
        return None
    return (
        cast("Direction | None", row.view_override_dir),
        cast("IVRegime | None", row.view_override_iv),
    )
