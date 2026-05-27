"""Tests for latest_snapshot + score_breakdown (IBK-56, IBK-57)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import insert

from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.tools.snapshots import register
from optionsbot.storage.schema import snapshots, strategy_scores
from tests.unit.test_mcp.conftest import FakeCtx, get_tools


def _seed_snapshot_with_scores(
    server_context: ServerContext,
    *,
    symbol: str = "SPY",
    ts: datetime | None = None,
) -> int:
    ts = ts or datetime.now(UTC)
    with server_context.engine.begin() as conn:
        result = conn.execute(insert(snapshots).values(
            symbol=symbol, ts=ts, spot=400.0,
            iv_rank=0.7, hv20=0.2, iv_hv_ratio=1.1,
            regime_dir="neutral", regime_iv="high",
            raw_json={"warming_up": False},
        ))
        snap_id = result.inserted_primary_key[0]
        conn.execute(insert(strategy_scores), [
            {
                "snapshot_id": snap_id, "strategy": "iron_condor",
                "score": 85.0, "rationale": "IC rationale",
                "legs_json": [{"symbol": symbol, "side": "sell", "strike": 410}],
            },
            {
                "snapshot_id": snap_id, "strategy": "iron_butterfly",
                "score": 65.0, "rationale": "IB rationale",
                "legs_json": [],
            },
        ])
    return snap_id


# ---- latest_snapshot (IBK-56) ---------------------------------------------

async def test_snapshot_ts_round_trips_with_utc_offset(
    server_context: ServerContext,
) -> None:
    """SQLite strips tz info from DateTime(tz=True) reads. Both snapshot tools
    must run timestamps through iso_utc() so callers always see +00:00."""
    _seed_snapshot_with_scores(server_context)
    tools = get_tools(register)
    latest = tools["latest_snapshot"]
    breakdown = tools["score_breakdown"]

    latest_result = await latest(symbol="SPY", ctx=FakeCtx(server_context))
    assert latest_result["snapshot"]["ts"].endswith("+00:00")

    breakdown_result = await breakdown(
        symbol="SPY", strategy="iron_condor", ctx=FakeCtx(server_context)
    )
    assert breakdown_result["snapshot_ts"].endswith("+00:00")


async def test_latest_snapshot_returns_latest_with_all_scores(
    server_context: ServerContext,
) -> None:
    older = datetime(2026, 5, 26, tzinfo=UTC)
    newer = datetime(2026, 5, 27, tzinfo=UTC)
    _seed_snapshot_with_scores(server_context, ts=older)
    snap_id = _seed_snapshot_with_scores(server_context, ts=newer)
    tools = get_tools(register)
    latest = tools["latest_snapshot"]

    result = await latest(symbol="SPY", ctx=FakeCtx(server_context))
    assert result["ok"] is True
    assert result["snapshot"]["id"] == snap_id
    assert result["snapshot"]["spot"] == 400.0
    # Returns ALL strategy scores, not just top-K.
    names = {s["strategy"] for s in result["strategies"]}
    assert names == {"iron_condor", "iron_butterfly"}


async def test_latest_snapshot_unknown_symbol_returns_error(
    server_context: ServerContext,
) -> None:
    tools = get_tools(register)
    latest = tools["latest_snapshot"]
    result = await latest(symbol="NOPE", ctx=FakeCtx(server_context))
    assert result == {
        "ok": False,
        "error": "not_found",
        "message": "no snapshot for NOPE",
        "symbol": "NOPE",
    }


async def test_latest_snapshot_includes_raw_json(
    server_context: ServerContext,
) -> None:
    _seed_snapshot_with_scores(server_context)
    tools = get_tools(register)
    latest = tools["latest_snapshot"]
    result = await latest(symbol="SPY", ctx=FakeCtx(server_context))
    assert result["snapshot"]["raw_json"] == {"warming_up": False}


# ---- score_breakdown (IBK-57) ---------------------------------------------

async def test_score_breakdown_returns_named_strategy(
    server_context: ServerContext,
) -> None:
    _seed_snapshot_with_scores(server_context)
    tools = get_tools(register)
    breakdown = tools["score_breakdown"]
    result = await breakdown(symbol="SPY", strategy="iron_condor", ctx=FakeCtx(server_context))

    assert result["ok"] is True
    assert result["strategy"] == "iron_condor"
    assert result["score"] == 85.0
    assert result["rationale"] == "IC rationale"
    assert result["legs"] == [{"symbol": "SPY", "side": "sell", "strike": 410}]


async def test_score_breakdown_unknown_symbol(
    server_context: ServerContext,
) -> None:
    tools = get_tools(register)
    breakdown = tools["score_breakdown"]
    result = await breakdown(symbol="NOPE", strategy="iron_condor", ctx=FakeCtx(server_context))
    assert result["ok"] is False
    assert result["error"] == "not_found"


async def test_score_breakdown_unknown_strategy(
    server_context: ServerContext,
) -> None:
    _seed_snapshot_with_scores(server_context)
    tools = get_tools(register)
    breakdown = tools["score_breakdown"]
    result = await breakdown(symbol="SPY", strategy="moonshot", ctx=FakeCtx(server_context))
    assert result["ok"] is False
    assert result["error"] == "not_found"
    assert "no score for strategy moonshot" in result["message"]


async def test_score_breakdown_uses_latest_snapshot(
    server_context: ServerContext,
) -> None:
    """Score breakdown reads from the LATEST snapshot's scores, not an older one."""
    older = datetime(2026, 5, 26, tzinfo=UTC)
    newer = datetime(2026, 5, 27, tzinfo=UTC)
    with server_context.engine.begin() as conn:
        result = conn.execute(insert(snapshots).values(
            symbol="SPY", ts=older, spot=399.0,
            regime_dir="neutral", regime_iv="high",
        ))
        old_id = result.inserted_primary_key[0]
        conn.execute(insert(strategy_scores).values(
            snapshot_id=old_id, strategy="iron_condor", score=50.0,
            rationale="old", legs_json=[],
        ))
    _seed_snapshot_with_scores(server_context, ts=newer)
    tools = get_tools(register)
    breakdown = tools["score_breakdown"]
    result = await breakdown(symbol="SPY", strategy="iron_condor", ctx=FakeCtx(server_context))
    assert result["score"] == 85.0
    assert result["rationale"] == "IC rationale"
