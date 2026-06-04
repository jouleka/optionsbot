"""Tests for run_scan_tick."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import insert, select

from optionsbot.analysis.types import MarketView
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.scan_runner import run_scan_tick
from optionsbot.scan.types import ScanResult
from optionsbot.storage.schema import scan_runs, watchlist


def _fake_scan_result(symbol: str = "SPY") -> ScanResult:
    view = MarketView(
        direction="neutral", direction_strength="weak", iv_regime="high",
        iv_rank_value=0.7, earnings_in_window=False, warming_up=False,
    )
    return ScanResult(
        symbol=symbol, snapshot_id=42,
        snapshot_ts=datetime(2026, 5, 27, 15, 30, tzinfo=UTC),
        view=view, scored=(),
    )


def _scan_result_for(symbol, scored) -> ScanResult:
    view = MarketView(
        direction="neutral", direction_strength="weak", iv_regime="high",
        iv_rank_value=0.7, earnings_in_window=False, warming_up=False,
    )
    return ScanResult(
        symbol=symbol, snapshot_id=7,
        snapshot_ts=datetime(2026, 6, 2, 15, 30, tzinfo=UTC),
        view=view, scored=scored,
    )


async def test_run_scan_tick_short_circuits_when_market_closed(
    daemon_context: DaemonContext,
) -> None:
    with patch(
        "optionsbot.daemon.scan_runner.is_market_open", return_value=False
    ):
        summary = await run_scan_tick(daemon_context)
    assert summary.tickers_scanned == 0
    assert summary.alerts_enqueued == 0


async def test_run_scan_tick_scans_each_watchlist_symbol(
    daemon_context: DaemonContext,
) -> None:
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=datetime.now(UTC)))
        conn.execute(insert(watchlist).values(symbol="MSFT", added_at=datetime.now(UTC)))

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), \
         patch(
            "optionsbot.daemon.scan_runner.scan_symbol",
            new=AsyncMock(side_effect=lambda s, *a, **kw: _fake_scan_result(s)),
         ) as mock_scan:
        summary = await run_scan_tick(daemon_context)

    assert summary.tickers_scanned == 2
    assert mock_scan.await_count == 2
    for call in mock_scan.await_args_list:
        assert call.kwargs["resolver"] is daemon_context.resolver


async def test_run_scan_tick_passes_view_override_from_watchlist(
    daemon_context: DaemonContext,
) -> None:
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(
            symbol="AAPL", added_at=datetime.now(UTC),
            view_override_dir="bull", view_override_iv="high",
        ))

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), \
         patch(
            "optionsbot.daemon.scan_runner.scan_symbol",
            new=AsyncMock(return_value=_fake_scan_result("AAPL")),
         ) as mock_scan:
        await run_scan_tick(daemon_context)

    call_kwargs = mock_scan.await_args.kwargs
    assert call_kwargs["view_override"] == ("bull", "high")


async def test_run_scan_tick_persists_scan_runs_row(
    daemon_context: DaemonContext,
) -> None:
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=datetime.now(UTC)))

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), \
         patch(
            "optionsbot.daemon.scan_runner.scan_symbol",
            new=AsyncMock(return_value=_fake_scan_result("AAPL")),
         ):
        await run_scan_tick(daemon_context)

    with daemon_context.engine.connect() as conn:
        rows = conn.execute(select(scan_runs)).fetchall()
    assert len(rows) == 1
    assert rows[0].tickers_scanned == 1
    assert rows[0].finished is not None


async def test_run_scan_tick_records_per_symbol_errors_without_aborting_tick(
    daemon_context: DaemonContext,
) -> None:
    """One symbol failing should not stop the rest of the watchlist."""
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=datetime.now(UTC)))
        conn.execute(insert(watchlist).values(symbol="BORK", added_at=datetime.now(UTC)))
        conn.execute(insert(watchlist).values(symbol="MSFT", added_at=datetime.now(UTC)))

    async def fake_scan(symbol, *a, **kw):
        if symbol == "BORK":
            raise ValueError("simulated chain fetch failure")
        return _fake_scan_result(symbol)

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), \
         patch("optionsbot.daemon.scan_runner.scan_symbol", new=AsyncMock(side_effect=fake_scan)):
        summary = await run_scan_tick(daemon_context)

    assert summary.tickers_scanned == 2  # AAPL + MSFT
    assert len(summary.errors) == 1
    assert "BORK" in summary.errors[0]


async def test_run_scan_tick_enqueues_top_n_above_floor(
    daemon_context: DaemonContext,
) -> None:
    """The top alert_top_n (default 3) floor-passing picks are enqueued; lower
    ones are dropped by the top-N cap."""
    from optionsbot.scoring import ScoredStrategy
    from optionsbot.scoring.types import FactorBreakdown

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
            strategy_name=name, score=score,
            factors=FactorBreakdown(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            suggestion=sug, rationale="...",
        )

    scored = (
        _make_scored("iron_condor", 90.0),
        _make_scored("iron_butterfly", 85.0),
        _make_scored("bull_put_spread", 72.0),
        _make_scored("calendar_spread", 65.0),  # 4th-best -> dropped by the top-3 cap
    )
    result = _fake_scan_result("SPY")
    result = ScanResult(
        symbol="SPY", snapshot_id=99,
        snapshot_ts=result.snapshot_ts, view=result.view,
        scored=scored,
    )
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="SPY", added_at=datetime.now(UTC)))

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), \
         patch(
            "optionsbot.daemon.scan_runner.scan_symbol",
            new=AsyncMock(return_value=result),
         ), \
         patch(
            "optionsbot.daemon.scan_runner.enqueue_alert",
            new=AsyncMock(),
         ) as mock_enqueue:
        summary = await run_scan_tick(daemon_context)

    assert mock_enqueue.await_count == 3
    assert summary.alerts_enqueued == 3


def test_scan_settings_alert_calibration_defaults() -> None:
    from optionsbot.config import ScanSettings

    s = ScanSettings()
    assert s.alert_top_n == 3
    assert s.score_threshold == 55  # repurposed as the alert quality floor


def test_rank_alert_candidates_floors_and_sorts() -> None:
    from optionsbot.daemon.scan_runner import rank_alert_candidates

    picks = [
        ("SPY", MagicMock(score=60.0), 1),
        ("AAPL", MagicMock(score=80.0), 2),
        ("XYZ", MagicMock(score=40.0), 3),  # below floor -> dropped
    ]
    out = rank_alert_candidates(picks, score_floor=50.0)
    assert [(sym, p.score) for sym, p, _ in out] == [("AAPL", 80.0), ("SPY", 60.0)]


async def test_run_scan_tick_alerts_top_n_across_all_symbols(
    daemon_context: DaemonContext,
) -> None:
    """top_n is applied across the WHOLE tick, not per symbol: the 2 highest of
    4 floor-passing picks (across 2 symbols) are enqueued."""
    from optionsbot.scoring import ScoredStrategy
    from optionsbot.scoring.types import FactorBreakdown

    def _mk(name: str, score: float) -> ScoredStrategy:
        sug = MagicMock()
        sug.legs = ()
        sug.prob_profit = 0.6
        return ScoredStrategy(
            strategy_name=name, score=score,
            factors=FactorBreakdown(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            suggestion=sug, rationale="...",
        )

    daemon_context.settings.scan.alert_top_n = 2
    daemon_context.settings.scan.score_threshold = 50
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="SPY", added_at=datetime.now(UTC)))
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=datetime.now(UTC)))

    spy = _scan_result_for("SPY", (_mk("a", 88.0), _mk("b", 60.0)))
    aapl = _scan_result_for("AAPL", (_mk("c", 75.0), _mk("d", 55.0)))

    async def fake_scan(symbol, *a, **kw):
        return spy if symbol == "SPY" else aapl

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), patch(
        "optionsbot.daemon.scan_runner.scan_symbol", new=AsyncMock(side_effect=fake_scan)
    ), patch(
        "optionsbot.daemon.scan_runner.enqueue_alert", new=AsyncMock(return_value=True)
    ) as mock_enqueue:
        summary = await run_scan_tick(daemon_context)

    assert mock_enqueue.await_count == 2
    assert summary.alerts_enqueued == 2
    enqueued = sorted(call.args[2].score for call in mock_enqueue.await_args_list)
    assert enqueued == [75.0, 88.0]  # the 2 best across BOTH symbols (not 60/55)
