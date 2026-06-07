"""Tests for the daily_brief MCP tool (IBK-107)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import insert

from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.tools.daily_brief import (
    _edge_tier,
    _reconstruct_suggestion,
    register,
)
from optionsbot.storage.schema import snapshots, strategy_scores, symbol_news, watchlist
from tests.unit.test_mcp.conftest import FakeCtx, get_tools


def test_reconstruct_suggestion_enables_canonical_edge() -> None:
    sj = {
        "expected_value": -49.0, "max_loss": 737.0, "credit_or_debit": 2.6,
        "max_profit": 263.0, "prob_profit": 0.67, "suggested_quantity": 1,
        "defined_risk": True, "reward_risk": 0.36, "risk_tier": "balanced",
    }
    sug = _reconstruct_suggestion(sj, "bull_put_spread", "ok")
    assert sug.expected_value == -49.0
    assert sug.max_loss == 737.0
    # The canonical property recomputes from the reconstructed fields:
    assert sug.risk_normalized_expectancy == -49.0 / 737.0
    assert _edge_tier(sug) == "negative"


def test_edge_tier_mapping() -> None:
    positive = _reconstruct_suggestion({"expected_value": 5.0, "max_loss": 100.0}, "x", "")
    none_edge = _reconstruct_suggestion({"expected_value": 5.0, "max_loss": None}, "x", "")
    break_even = _reconstruct_suggestion({"expected_value": 0.0, "max_loss": 100.0}, "x", "")
    assert _edge_tier(positive) == "positive"
    assert _edge_tier(none_edge) == "undefined"
    assert _edge_tier(break_even) == "negative"


def _seed_symbol(conn, symbol, *, setups, regime_dir="neutral", regime_iv="high", iv_rank=0.7):
    """Seed one snapshot + its strategy_scores. setups: list of (strategy, score, sj)."""
    snap_id = conn.execute(
        insert(snapshots).values(
            symbol=symbol, ts=datetime(2026, 6, 5, 20, 0, tzinfo=UTC),
            regime_dir=regime_dir, regime_iv=regime_iv, iv_rank=iv_rank,
        )
    ).inserted_primary_key[0]
    for strategy, score, sj in setups:
        conn.execute(insert(strategy_scores).values(
            snapshot_id=snap_id, strategy=strategy, score=score,
            rationale="ok", legs_json=[], suggestion_json=sj,
        ))


async def test_daily_brief_ranks_positive_edge_symbol_first(
    server_context: ServerContext,
) -> None:
    with server_context.engine.begin() as conn:
        _seed_symbol(conn, "NVDA", regime_dir="bull", iv_rank=0.76, setups=[
            ("cash_secured_put", 64.5, {"expected_value": -83.0, "max_loss": 19397.0}),
        ])
        _seed_symbol(conn, "AAPL", setups=[
            ("bull_put_spread", 72.0, {"expected_value": 12.0, "max_loss": 600.0}),
        ])
    brief = get_tools(register)["daily_brief"]

    result = await brief(symbols=["NVDA", "AAPL"], ctx=FakeCtx(server_context))

    assert result["ok"] is True
    assert result["any_positive_edge"] is True
    # AAPL has a +EV setup -> ranks above NVDA (only -EV).
    assert [e["symbol"] for e in result["ranked"]] == ["AAPL", "NVDA"]
    assert result["ranked"][0]["top_setups"][0]["edge_tier"] == "positive"
    assert result["ranked"][1]["no_positive_edge"] is True
    assert result["rubric"].startswith("You are reasoning over")


async def test_daily_brief_all_negative_is_honest_and_raw_ev_ordered(
    server_context: ServerContext,
) -> None:
    with server_context.engine.begin() as conn:
        _seed_symbol(conn, "NVDA", regime_dir="bull", iv_rank=0.76, setups=[
            ("cash_secured_put", 64.5, {"expected_value": -83.0, "max_loss": 19397.0}),
            ("bull_put_spread", 60.8, {"expected_value": -49.0, "max_loss": 737.0}),
        ])
    brief = get_tools(register)["daily_brief"]

    result = await brief(symbols=["NVDA"], ctx=FakeCtx(server_context))

    assert result["any_positive_edge"] is False
    setups = result["ranked"][0]["top_setups"]
    # -EV group orders by RAW EV: spread (-49) above CSP (-83) -- the IBK-106 rule.
    assert [s["strategy"] for s in setups] == ["bull_put_spread", "cash_secured_put"]
    assert all(s["edge_tier"] == "negative" for s in setups)


async def test_daily_brief_defaults_to_watchlist_and_notes_unscanned(
    server_context: ServerContext,
) -> None:
    with server_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="SPY", added_at=datetime.now(UTC)))
        # SPY has no snapshot row.
    brief = get_tools(register)["daily_brief"]

    result = await brief(symbols=None, ctx=FakeCtx(server_context))  # None -> watchlist

    assert result["generated_for"] == ["SPY"]
    assert result["ranked"] == []
    assert any("no data for SPY" in n for n in result["notes"])


async def test_daily_brief_dedupes_case_variant_symbols(
    server_context: ServerContext,
) -> None:
    with server_context.engine.begin() as conn:
        _seed_symbol(conn, "AAPL", setups=[
            ("bull_put_spread", 72.0, {"expected_value": 12.0, "max_loss": 600.0}),
        ])
    brief = get_tools(register)["daily_brief"]

    result = await brief(symbols=["AAPL", "aapl", "AAPL"], ctx=FakeCtx(server_context))

    assert result["generated_for"] == ["AAPL"]   # upper-cased + de-duplicated
    assert len(result["ranked"]) == 1


async def test_daily_brief_surfaces_headlines_and_earnings(
    server_context: ServerContext,
) -> None:
    with server_context.engine.begin() as conn:
        snap_id = conn.execute(insert(snapshots).values(
            symbol="NVDA", ts=datetime(2026, 6, 5, 20, 0, tzinfo=UTC),
            regime_dir="bull", regime_iv="high", iv_rank=0.76,
            raw_json={"earnings_in_window": True},
        )).inserted_primary_key[0]
        conn.execute(insert(strategy_scores).values(
            snapshot_id=snap_id, strategy="bull_put_spread", score=72.0,
            rationale="ok", legs_json=[],
            suggestion_json={"expected_value": 10.0, "max_loss": 600.0}))
        conn.execute(insert(symbol_news).values(
            symbol="NVDA", fetched_at=datetime.now(UTC),
            headlines_json=[{"title": "NVDA upgraded", "publisher": "Reuters",
                             "published_ts": None, "link": "https://r/1"}]))
    brief = get_tools(register)["daily_brief"]

    result = await brief(symbols=["NVDA"], ctx=FakeCtx(server_context))

    entry = result["ranked"][0]
    assert entry["earnings_in_window"] is True
    assert entry["headlines"][0]["title"] == "NVDA upgraded"


async def test_daily_brief_headlines_empty_when_no_news(
    server_context: ServerContext,
) -> None:
    with server_context.engine.begin() as conn:
        snap_id = conn.execute(insert(snapshots).values(
            symbol="AAPL", ts=datetime(2026, 6, 5, 20, 0, tzinfo=UTC),
            regime_dir="neutral", regime_iv="high", iv_rank=0.5, raw_json={},
        )).inserted_primary_key[0]
        conn.execute(insert(strategy_scores).values(
            snapshot_id=snap_id, strategy="bull_put_spread", score=72.0,
            rationale="ok", legs_json=[],
            suggestion_json={"expected_value": 10.0, "max_loss": 600.0}))
    brief = get_tools(register)["daily_brief"]

    result = await brief(symbols=["AAPL"], ctx=FakeCtx(server_context))

    entry = result["ranked"][0]
    assert entry["headlines"] == []
    assert entry["earnings_in_window"] is None  # not persisted on this snapshot
