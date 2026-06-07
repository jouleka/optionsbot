"""daily_brief tool (IBK-107).

Read-only cross-symbol synthesis: assembles a grounded "what makes most sense
today" decision packet from the latest PERSISTED snapshots (no IBKR, no LLM, no
writes) for a Claude MCP client to reason over. Edge is reconstructed from each
persisted ``suggestion_json`` so the canonical ``edge_sort_key`` /
``has_positive_edge`` apply unchanged -- the brief can never disagree with /scan.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from sqlalchemy import desc, select

from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.serialization import iso_utc
from optionsbot.scoring import DEFAULT_TOP_K
from optionsbot.scoring.composite import edge_sort_key, has_positive_edge
from optionsbot.storage.schema import snapshots, strategy_scores, watchlist
from optionsbot.strategies.base import StrategySuggestion

_TIER_NAMES = {2: "positive", 1: "negative", 0: "undefined"}

RUBRIC = (
    "You are reasoning over a grounded cross-symbol options brief. Rules:\n"
    "1. If any_positive_edge is false, say plainly that nothing has positive edge "
    "today and do NOT manufacture a pick; you may mention the least-bad setup for "
    "reference, clearly labeled not a recommendation.\n"
    "2. Otherwise lead with the single best risk-adjusted setup -- the first ranked "
    "entry whose top setup has edge_tier 'positive' -- and state its symbol, strategy, "
    "expected_value, prob_profit, max_loss, and why.\n"
    "3. Offer one higher-reward alternative ONLY if it also has edge_tier 'positive'.\n"
    "4. Flag any stale snapshot_ts. Earnings proximity is NOT in this packet -- remind "
    "the user to check the earnings calendar before trading.\n"
    "5. Reason ONLY over the numbers in this packet; never invent expected_value, "
    "prob_profit, or edge."
)


def _reconstruct_suggestion(
    suggestion_json: dict[str, Any] | None, strategy_name: str, rationale: str
) -> StrategySuggestion:
    """Rebuild a StrategySuggestion from a persisted strategy_scores row.

    ``legs=()`` -- the edge math ignores legs; the raw ``legs_json`` is carried
    separately in the packet for display. Reusing the real class means
    ``risk_normalized_expectancy`` / ``edge_sort_key`` / ``has_positive_edge``
    apply canonically (no edge-formula duplication).
    """
    sj = suggestion_json or {}
    return StrategySuggestion(
        strategy_name=strategy_name,
        legs=(),
        credit_or_debit=sj.get("credit_or_debit", 0.0),
        max_loss=sj.get("max_loss"),
        max_profit=sj.get("max_profit"),
        prob_profit=sj.get("prob_profit"),
        suggested_quantity=sj.get("suggested_quantity", 0),
        defined_risk=sj.get("defined_risk", True),
        rationale=rationale,
        reward_risk=sj.get("reward_risk"),
        expected_value=sj.get("expected_value"),
        risk_tier=sj.get("risk_tier", "balanced"),
    )


def _edge_tier(suggestion: StrategySuggestion) -> str:
    """Human-readable tier behind edge_sort_key: positive / negative / undefined."""
    return _TIER_NAMES[edge_sort_key(suggestion)[0]]


def _resolve_symbols(symbols: list[str] | None, lifespan: ServerContext) -> list[str]:
    """The given symbols (upper-cased), or the watchlist when ``symbols`` is None."""
    if symbols:
        return [s.upper().strip() for s in symbols]
    with lifespan.engine.connect() as conn:
        rows = conn.execute(
            select(watchlist.c.symbol).order_by(watchlist.c.symbol)
        ).fetchall()
    return [r.symbol for r in rows]


def _setup_dict(row: Any, sug: StrategySuggestion) -> dict[str, Any]:
    return {
        "strategy": row.strategy,
        "score": row.score,
        "prob_profit": sug.prob_profit,
        "expected_value": sug.expected_value,
        "reward_risk": sug.reward_risk,
        "max_loss": sug.max_loss,
        "risk_tier": sug.risk_tier,
        "edge_tier": _edge_tier(sug),
        "legs": row.legs_json or [],
    }


def _assemble_brief(symbols: list[str], lifespan: ServerContext) -> dict[str, Any]:
    scored_entries: list[tuple[tuple[int, float], dict[str, Any]]] = []
    notes: list[str] = []
    any_positive = False
    with lifespan.engine.connect() as conn:
        for symbol in symbols:
            snap = conn.execute(
                select(snapshots)
                .where(snapshots.c.symbol == symbol)
                .order_by(desc(snapshots.c.ts))
                .limit(1)
            ).first()
            if snap is None:
                notes.append(f"no data for {symbol} (not scanned yet)")
                continue
            rows = conn.execute(
                select(strategy_scores).where(strategy_scores.c.snapshot_id == snap.id)
            ).fetchall()
            # (row, reconstructed suggestion) pairs, sign-aware best-edge first.
            pairs = sorted(
                (
                    (r, _reconstruct_suggestion(r.suggestion_json, r.strategy, r.rationale or ""))
                    for r in rows
                ),
                key=lambda p: edge_sort_key(p[1]),
                reverse=True,
            )
            symbol_positive = any(has_positive_edge(s) for _, s in pairs)
            any_positive = any_positive or symbol_positive
            best_key: tuple[int, float] = (
                edge_sort_key(pairs[0][1]) if pairs else (-1, float("-inf"))
            )
            entry: dict[str, Any] = {
                "symbol": symbol,
                "snapshot_ts": iso_utc(snap.ts),
                "view": {
                    "direction": snap.regime_dir,
                    "iv_regime": snap.regime_iv,
                    "iv_rank_value": snap.iv_rank,
                },
                "no_positive_edge": not symbol_positive,
                "top_setups": [_setup_dict(r, s) for r, s in pairs[:DEFAULT_TOP_K]],
            }
            scored_entries.append((best_key, entry))
    scored_entries.sort(key=lambda t: t[0], reverse=True)
    return {
        "ok": True,
        "generated_for": symbols,
        "any_positive_edge": any_positive,
        "ranked": [entry for _, entry in scored_entries],
        "notes": notes,
        "rubric": RUBRIC,
    }


def register(server: FastMCP) -> None:
    @server.tool()
    async def daily_brief(
        symbols: list[str] | None,
        ctx: Context[ServerSession, ServerContext],
    ) -> dict[str, Any]:
        """Cross-symbol "what makes most sense today" decision packet (read-only).

        Pass ``symbols`` as a list, or ``null`` to use the watchlist. Reads the
        latest persisted snapshot per symbol (no live IBKR scan) and returns a
        sign-aware, best-edge-first packet plus a reasoning ``rubric``. Reason
        ONLY over the returned numbers.
        """
        lifespan = ctx.request_context.lifespan_context
        resolved = _resolve_symbols(symbols, lifespan)
        if not resolved:
            return {
                "ok": True,
                "generated_for": [],
                "any_positive_edge": False,
                "ranked": [],
                "notes": ["watchlist is empty -- pass symbols=[...] or add to the watchlist"],
                "rubric": RUBRIC,
            }
        return _assemble_brief(resolved, lifespan)
