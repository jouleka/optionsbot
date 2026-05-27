"""Tests for the analyze MCP tool (IBK-54)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import insert

from optionsbot.analysis.types import MarketView
from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.tools.analyze import register
from optionsbot.scan.types import ScanResult
from optionsbot.scoring import ScoredStrategy
from optionsbot.scoring.types import FactorBreakdown
from optionsbot.storage.schema import snapshots, strategy_scores, watchlist

# Import the consolidated test helpers
from tests.unit.test_mcp.conftest import FakeCtx, get_tools


def _fake_scan_result(symbol: str = "SPY") -> ScanResult:
    view = MarketView(
        direction="neutral",
        direction_strength="weak",
        iv_regime="high",
        iv_rank_value=0.7,
        earnings_in_window=False,
        warming_up=False,
    )
    return ScanResult(
        symbol=symbol,
        snapshot_id=42,
        snapshot_ts=datetime(2026, 5, 27, 15, 30, tzinfo=UTC),
        view=view,
        scored=(),
    )


async def test_analyze_fresh_invokes_scan_symbol(
    server_context: ServerContext, mock_ibkr_client: MagicMock
) -> None:
    server_context._ibkr = mock_ibkr_client
    tools = get_tools(register)
    analyze = tools["analyze"]

    with patch(
        "optionsbot.mcp_server.tools.analyze.scan_symbol",
        new=AsyncMock(return_value=_fake_scan_result()),
    ) as mock_scan:
        result = await analyze(symbol="SPY", fresh=True, ctx=FakeCtx(server_context))

    mock_scan.assert_awaited_once()
    assert result["ok"] is True
    assert result["symbol"] == "SPY"
    assert result["snapshot_id"] == 42
    assert result["view"]["direction"] == "neutral"


async def test_analyze_fresh_applies_view_override_from_watchlist(
    server_context: ServerContext, mock_ibkr_client: MagicMock
) -> None:
    server_context._ibkr = mock_ibkr_client
    with server_context.engine.begin() as conn:
        conn.execute(
            insert(watchlist).values(
                symbol="SPY", added_at=datetime.now(UTC), view_override_dir="bull"
            )
        )
    tools = get_tools(register)
    analyze = tools["analyze"]

    with patch(
        "optionsbot.mcp_server.tools.analyze.scan_symbol",
        new=AsyncMock(return_value=_fake_scan_result()),
    ) as mock_scan:
        await analyze(symbol="SPY", fresh=True, ctx=FakeCtx(server_context))

    call_kwargs = mock_scan.await_args.kwargs
    assert call_kwargs["view_override"] == ("bull", None)


async def test_analyze_cached_reads_latest_snapshot(
    server_context: ServerContext,
) -> None:
    """When fresh=False and a snapshot exists, return it without calling scan."""
    older = datetime(2026, 5, 26, tzinfo=UTC)
    newer = datetime(2026, 5, 27, tzinfo=UTC)
    with server_context.engine.begin() as conn:
        conn.execute(
            insert(snapshots).values(
                symbol="SPY", ts=older, spot=399.0, regime_dir="neutral", regime_iv="high"
            )
        )
        result = conn.execute(
            insert(snapshots).values(
                symbol="SPY", ts=newer, spot=401.0, regime_dir="bull", regime_iv="high"
            )
        )
        snap_id = result.inserted_primary_key[0]
        conn.execute(
            insert(strategy_scores).values(
                snapshot_id=snap_id,
                strategy="iron_condor",
                score=82.0,
                rationale="...",
                legs_json=[],
            )
        )
    tools = get_tools(register)
    analyze = tools["analyze"]

    with patch(
        "optionsbot.mcp_server.tools.analyze.scan_symbol",
    ) as mock_scan:
        result = await analyze(symbol="SPY", fresh=False, ctx=FakeCtx(server_context))

    mock_scan.assert_not_called()
    assert result["ok"] is True
    assert result["snapshot_id"] == snap_id
    assert result["view"]["direction"] == "bull"
    assert len(result["top_strategies"]) == 1
    assert result["top_strategies"][0]["strategy_name"] == "iron_condor"


async def test_analyze_cached_empty_returns_no_snapshot_error(
    server_context: ServerContext,
) -> None:
    tools = get_tools(register)
    analyze = tools["analyze"]
    result = await analyze(symbol="SPY", fresh=False, ctx=FakeCtx(server_context))
    assert result["ok"] is False
    assert result["error"] == "no_snapshot"
    assert "retry with fresh=true" in result["hint"]


async def test_analyze_returns_top_k_only(
    server_context: ServerContext, mock_ibkr_client: MagicMock
) -> None:
    """analyze returns top-K (default 3) of scored strategies above threshold."""
    server_context._ibkr = mock_ibkr_client

    def _make_scored(name: str, score: float) -> ScoredStrategy:
        sug = MagicMock()
        sug.legs = ()
        sug.credit_or_debit = 0.0
        sug.max_loss = 0.0
        sug.max_profit = 0.0
        sug.prob_profit = 0.5
        sug.suggested_quantity = 1
        sug.defined_risk = True
        return ScoredStrategy(
            strategy_name=name,
            score=score,
            factors=FactorBreakdown(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            suggestion=sug,
            rationale="...",
        )

    scored = (
        _make_scored("iron_condor", 90.0),
        _make_scored("iron_butterfly", 85.0),
        _make_scored("bull_put_spread", 80.0),
        _make_scored("calendar_spread", 75.0),
        _make_scored("diagonal_spread", 65.0),
    )
    fake_result_base = _fake_scan_result()
    fake_result = ScanResult(
        symbol="SPY",
        snapshot_id=99,
        snapshot_ts=fake_result_base.snapshot_ts,
        view=fake_result_base.view,
        scored=scored,
    )
    tools = get_tools(register)
    analyze = tools["analyze"]

    with patch(
        "optionsbot.mcp_server.tools.analyze.scan_symbol",
        new=AsyncMock(return_value=fake_result),
    ):
        result = await analyze(symbol="SPY", fresh=True, ctx=FakeCtx(server_context))

    names = [s["strategy_name"] for s in result["top_strategies"]]
    assert names == ["iron_condor", "iron_butterfly", "bull_put_spread"]
