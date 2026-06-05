"""Tests for run_scan_tick."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import insert, select

from optionsbot.analysis.types import MarketView
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.scan_runner import _resolve_scan_symbols, run_scan_tick
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
        sug.risk_normalized_expectancy = score / 1000.0
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

    def _pick(sym, score, rne):
        scored = MagicMock(score=score)
        scored.suggestion.risk_normalized_expectancy = rne
        return (sym, scored, 1)

    picks = [
        _pick("SPY", 60.0, 0.02),
        _pick("AAPL", 80.0, 0.10),
        _pick("XYZ", 40.0, 0.99),   # below floor -> dropped despite high edge
    ]
    out = rank_alert_candidates(picks, score_floor=50.0)
    # Survivors (score>=50) ordered by edge desc: AAPL(0.10) then SPY(0.02).
    assert [sym for sym, _, _ in out] == ["AAPL", "SPY"]


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
        sug.risk_normalized_expectancy = score / 1000.0
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


async def test_resolve_scan_symbols_watchlist_only_when_auto_screen_off(
    daemon_context: DaemonContext,
) -> None:
    """auto_screen off -> exactly the watchlist; screen_universe is never called."""
    daemon_context.settings.scan.auto_screen = False
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(
            symbol="AAPL", added_at=datetime.now(UTC),
            view_override_dir="bull", view_override_iv="high",
        ))

    boom = AsyncMock(side_effect=AssertionError("screen_universe must not be called"))
    with patch("optionsbot.daemon.scan_runner.screen_universe", new=boom):
        resolved = await _resolve_scan_symbols(daemon_context)

    assert resolved == [("AAPL", ("bull", "high"))]
    boom.assert_not_awaited()


async def test_resolve_scan_symbols_unions_screened_with_watchlist(
    daemon_context: DaemonContext,
) -> None:
    """auto_screen on -> watchlist union top screened, deduped by symbol;
    watchlist override wins on overlap; screened-only names get override None."""
    from optionsbot.screener.screen import ScreenCandidate

    daemon_context.settings.scan.auto_screen = True
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(
            symbol="SPY", added_at=datetime.now(UTC),
            view_override_dir="bull", view_override_iv="high",
        ))
        conn.execute(insert(watchlist).values(
            symbol="AAPL", added_at=datetime.now(UTC),
            view_override_dir="bear", view_override_iv="low",
        ))

    candidates = (
        ScreenCandidate(symbol="NVDA", hv_rank=0.9, dollar_volume=2e9),
        ScreenCandidate(symbol="AAPL", hv_rank=0.8, dollar_volume=1e9),  # dup of watchlist
        ScreenCandidate(symbol="AMD", hv_rank=0.7, dollar_volume=5e8),
    )
    with patch(
        "optionsbot.daemon.scan_runner.screen_universe",
        new=AsyncMock(return_value=candidates),
    ):
        resolved = await _resolve_scan_symbols(daemon_context)

    # Robust to watchlist row ordering: check membership/dedup via a dict + length.
    assert len(resolved) == 4  # SPY, AAPL, NVDA, AMD -- AAPL appears once
    assert dict(resolved) == {
        "SPY": ("bull", "high"),   # watchlist override preserved
        "AAPL": ("bear", "low"),   # watchlist override wins over the screened dup
        "NVDA": None,              # screened-only
        "AMD": None,               # screened-only
    }


async def test_resolve_scan_symbols_falls_back_to_watchlist_when_screen_raises(
    daemon_context: DaemonContext,
) -> None:
    """If screening raises, fall back to the watchlist alone (never abort)."""
    daemon_context.settings.scan.auto_screen = True
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=datetime.now(UTC)))

    with patch(
        "optionsbot.daemon.scan_runner.screen_universe",
        new=AsyncMock(side_effect=RuntimeError("ibkr down")),
    ):
        resolved = await _resolve_scan_symbols(daemon_context)

    assert resolved == [("AAPL", None)]


async def test_run_scan_tick_scans_screened_and_watchlist_symbols(
    daemon_context: DaemonContext,
) -> None:
    """auto_screen on: run_scan_tick scans the watchlist union the screened
    top-K, and scans screened-only names with view_override=None."""
    from optionsbot.screener.screen import ScreenCandidate

    daemon_context.settings.scan.auto_screen = True
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="SPY", added_at=datetime.now(UTC)))

    candidates = (
        ScreenCandidate(symbol="NVDA", hv_rank=0.9, dollar_volume=2e9),
        ScreenCandidate(symbol="AMD", hv_rank=0.7, dollar_volume=5e8),
    )
    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), \
         patch(
            "optionsbot.daemon.scan_runner.screen_universe",
            new=AsyncMock(return_value=candidates),
         ), \
         patch(
            "optionsbot.daemon.scan_runner.scan_symbol",
            new=AsyncMock(side_effect=lambda s, *a, **kw: _fake_scan_result(s)),
         ) as mock_scan:
        summary = await run_scan_tick(daemon_context)

    scanned = {call.args[0] for call in mock_scan.await_args_list}
    assert scanned == {"SPY", "NVDA", "AMD"}
    assert summary.tickers_scanned == 3
    overrides = {
        call.args[0]: call.kwargs["view_override"] for call in mock_scan.await_args_list
    }
    assert overrides["NVDA"] is None
    assert overrides["AMD"] is None


async def test_run_scan_tick_holds_ibkr_lock_during_scan(
    daemon_context: DaemonContext,
) -> None:
    """The scan section runs under context.ibkr_lock (so on-demand /scan can't
    race a scheduled tick for the market-data line)."""
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=datetime.now(UTC)))

    held = {"during": None}

    async def fake_scan(symbol, *a, **kw):
        held["during"] = daemon_context.ibkr_lock.locked()
        return _fake_scan_result(symbol)

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), \
         patch("optionsbot.daemon.scan_runner.scan_symbol", new=AsyncMock(side_effect=fake_scan)):
        await run_scan_tick(daemon_context)

    assert held["during"] is True  # the lock was held while scanning
    assert daemon_context.ibkr_lock.locked() is False  # released afterward


async def test_run_scan_tick_suppresses_alerts_when_paused(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.scoring import ScoredStrategy
    from optionsbot.scoring.types import FactorBreakdown

    sug = MagicMock()
    sug.legs = ()
    sug.prob_profit = 0.6
    scored = (ScoredStrategy("a", 90.0, FactorBreakdown(.5,.5,.5,.5,.5,.5), sug, "x"),)
    daemon_context.alerting_paused = True
    daemon_context.settings.scan.score_threshold = 50
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="SPY", added_at=datetime.now(UTC)))

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), \
         patch("optionsbot.daemon.scan_runner.scan_symbol",
               new=AsyncMock(return_value=_scan_result_for("SPY", scored))), \
         patch("optionsbot.daemon.scan_runner.enqueue_alert", new=AsyncMock()) as mock_enq:
        summary = await run_scan_tick(daemon_context)

    mock_enq.assert_not_awaited()       # paused → no enqueue
    assert summary.alerts_enqueued == 0
    assert summary.tickers_scanned == 1  # but scanning still happened
