"""Tests for run_scan_tick."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, date, datetime, timedelta
from unittest.mock import ANY, AsyncMock, MagicMock, patch

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


async def test_or_scan_reuses_intraday_bars_for_active_shadow_hypothesis(
    daemon_context: DaemonContext,
) -> None:
    daemon_context.settings.scan.auto_screen = False
    daemon_context.settings.scan.opening_range_fvg_enabled = True
    daemon_context.settings.execution.zero_dte_only = True
    with daemon_context.engine.begin() as conn:
        conn.execute(
            insert(watchlist).values(symbol="SPY", added_at=datetime.now(UTC))
        )
    now = datetime.now(UTC)
    hypothesis = MagicMock()
    hypothesis.session = now.date().isoformat()
    hypothesis.option_expiry = now.strftime("%Y%m%d")
    hypothesis.signal_at = now.replace(microsecond=0)
    hypothesis.thesis_expires_at = now.replace(microsecond=0) + timedelta(hours=1)
    hypothesis.generator = "failed_breakout_reversal"
    hypothesis.hypothesis_id = "shadow-failed-breakout"
    history = MagicMock()
    intraday = MagicMock(name="completed_intraday_bars")
    history.get_intraday_history = AsyncMock(return_value=intraday)

    with patch(
        "optionsbot.daemon.scan_runner.is_market_open", return_value=True
    ), patch(
        "optionsbot.daemon.scan_runner.HistoryClient", return_value=history
    ), patch(
        "optionsbot.daemon.scan_runner.detect_opening_range_fvg",
        return_value=None,
    ), patch(
        "optionsbot.daemon.scan_runner.generate_shadow_hypotheses",
        return_value=(hypothesis,),
    ) as generate, patch(
        "optionsbot.daemon.scan_runner.nyse_session_close_utc",
        return_value=now + timedelta(hours=2),
    ), patch(
        "optionsbot.daemon.scan_runner.nyse_session_date",
        return_value=now.date(),
    ), patch(
        "optionsbot.daemon.scan_runner.scan_symbol",
        new=AsyncMock(return_value=_fake_scan_result("SPY")),
    ) as scan:
        summary = await run_scan_tick(daemon_context)

    assert summary.tickers_scanned == 1
    history.get_intraday_history.assert_awaited_once()
    assert generate.call_args.args[0] is intraday
    scan.assert_awaited_once()
    assert scan.await_args.kwargs["opening_range_signal"] is None
    assert scan.await_args.kwargs["managed_hypotheses"] == (hypothesis,)


async def test_or_scan_avoids_chain_when_no_or_or_active_shadow_hypothesis(
    daemon_context: DaemonContext,
) -> None:
    daemon_context.settings.scan.auto_screen = False
    daemon_context.settings.scan.opening_range_fvg_enabled = True
    daemon_context.settings.execution.zero_dte_only = True
    with daemon_context.engine.begin() as conn:
        conn.execute(
            insert(watchlist).values(symbol="SPY", added_at=datetime.now(UTC))
        )
    history = MagicMock()
    history.get_intraday_history = AsyncMock(return_value=MagicMock())

    with patch(
        "optionsbot.daemon.scan_runner.is_market_open", return_value=True
    ), patch(
        "optionsbot.daemon.scan_runner.HistoryClient", return_value=history
    ), patch(
        "optionsbot.daemon.scan_runner.detect_opening_range_fvg",
        return_value=None,
    ), patch(
        "optionsbot.daemon.scan_runner.generate_shadow_hypotheses",
        return_value=(),
    ), patch(
        "optionsbot.daemon.scan_runner.scan_symbol",
        new=AsyncMock(),
    ) as scan:
        summary = await run_scan_tick(daemon_context)

    assert summary.tickers_scanned == 1
    history.get_intraday_history.assert_awaited_once()
    scan.assert_not_awaited()


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


async def test_managed_capture_registration_runs_before_any_alert_candidate(
    daemon_context: DaemonContext,
) -> None:
    """A no-edge/no-alert tick still hands its persisted snapshot to shadow capture."""
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="SPY", added_at=datetime.now(UTC)))

    with (
        patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True),
        patch(
            "optionsbot.daemon.scan_runner.scan_symbol",
            new=AsyncMock(return_value=_fake_scan_result("SPY")),
        ),
        patch(
            "optionsbot.daemon.managed_capture.register_snapshot_opportunities",
            return_value=1,
        ) as register,
        patch(
            "optionsbot.daemon.scan_runner.enqueue_alert",
            new=AsyncMock(),
        ) as enqueue,
    ):
        summary = await run_scan_tick(daemon_context)

    register.assert_called_once_with(
        daemon_context.engine,
        daemon_context.settings,
        42,
        decision_batch_id=ANY,
    )
    enqueue.assert_not_awaited()
    assert summary.alerts_enqueued == 0


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
        sug.max_loss = 100.0
        sug.max_profit = 0.0
        sug.prob_profit = 0.5
        sug.suggested_quantity = 1
        sug.defined_risk = True
        sug.risk_normalized_expectancy = score / 1000.0
        sug.expected_value = score        # positive -> has_positive_edge True
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

    # Task 6: PositionsClient must return a large USD equity so all picks pass
    # the single-trade-cap affordability gate (a $100 structure fits).
    from decimal import Decimal

    from optionsbot.ibkr.types import AccountSummary
    _fake_summary = AccountSummary(
        net_liquidation=Decimal("50000"), buying_power=None,
        available_funds=Decimal("50000"), currency="USD",
    )
    mock_pos = MagicMock()
    mock_pos.get_account_summary = AsyncMock(return_value=_fake_summary)

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), \
         patch(
            "optionsbot.daemon.scan_runner.scan_symbol",
            new=AsyncMock(return_value=result),
         ), \
         patch(
            "optionsbot.daemon.scan_runner.enqueue_alert",
            new=AsyncMock(),
         ) as mock_enqueue, \
         patch("optionsbot.daemon.scan_runner.PositionsClient", return_value=mock_pos):
        summary = await run_scan_tick(daemon_context)

    assert mock_enqueue.await_count == 3
    assert summary.alerts_enqueued == 3


async def test_auto_execution_receives_each_alerted_candidate_once(
    daemon_context: DaemonContext,
) -> None:
    """One delivered alert must never become two automatic entry attempts."""
    from decimal import Decimal

    from optionsbot.ibkr.types import AccountSummary
    from optionsbot.scoring import ScoredStrategy
    from optionsbot.scoring.types import FactorBreakdown

    suggestion = MagicMock(
        legs=(),
        credit_or_debit=-100.0,
        max_loss=100.0,
        max_profit=150.0,
        prob_profit=0.5,
        suggested_quantity=1,
        defined_risk=True,
        risk_normalized_expectancy=0.1,
        expected_value=10.0,
    )
    scored = ScoredStrategy(
        strategy_name="long_call",
        score=80.0,
        factors=FactorBreakdown(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        suggestion=suggestion,
        rationale="one exact candidate",
    )
    base = _fake_scan_result("SPY")
    result = ScanResult(
        symbol="SPY",
        snapshot_id=99,
        snapshot_ts=base.snapshot_ts,
        view=base.view,
        scored=(scored,),
    )
    daemon_context.settings.execution.mode = "auto"
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="SPY", added_at=datetime.now(UTC)))
    positions = MagicMock()
    positions.get_account_summary = AsyncMock(
        return_value=AccountSummary(
            net_liquidation=Decimal("50000"),
            buying_power=None,
            available_funds=Decimal("50000"),
            currency="USD",
        )
    )

    with (
        patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True),
        patch(
            "optionsbot.daemon.scan_runner.scan_symbol",
            new=AsyncMock(return_value=result),
        ),
        patch(
            "optionsbot.daemon.scan_runner.enqueue_alert",
            new=AsyncMock(return_value=True),
        ),
        patch("optionsbot.daemon.scan_runner.PositionsClient", return_value=positions),
        patch(
            "optionsbot.daemon.auto_executor.auto_execute_candidates",
            new=AsyncMock(return_value=1),
        ) as auto_execute,
    ):
        await run_scan_tick(daemon_context)

    candidates = auto_execute.await_args.args[1]
    assert len(candidates) == 1
    assert candidates[0][0] == "SPY"
    assert candidates[0][1].strategy_name == "long_call"


def test_scan_settings_alert_calibration_defaults() -> None:
    from optionsbot.config import ScanSettings

    s = ScanSettings()
    assert s.alert_top_n == 3
    assert s.score_threshold == 55  # repurposed as the alert quality floor
    # IBK-149 scan-resilience timeout defaults
    assert s.scan_symbol_timeout_s == 30.0
    assert s.screen_timeout_s == 60.0
    assert s.external_data_timeout_s == 5.0


def test_rank_alert_candidates_floors_and_sorts() -> None:
    from optionsbot.daemon.scan_runner import rank_alert_candidates

    def _pick(sym, score, rne):
        scored = MagicMock(score=score)
        scored.suggestion.risk_normalized_expectancy = rne
        scored.suggestion.expected_value = rne * 100.0   # same sign as rne
        # Task 6: new gate requires defined_risk=True and max_loss within cap.
        scored.suggestion.defined_risk = True
        scored.suggestion.max_loss = 100.0               # 100 <= 5000*0.15=750 -> passes
        return (sym, scored, 1)

    picks = [
        _pick("SPY", 60.0, 0.02),
        _pick("AAPL", 80.0, 0.10),
        _pick("XYZ", 40.0, 0.99),   # below floor -> dropped despite high edge
    ]
    # Task 6: pass account_value_usd + single_trade_cap_pct (required by new signature)
    out = rank_alert_candidates(
        picks, score_floor=50.0, account_value_usd=5000.0, single_trade_cap_pct=0.15
    )
    # Survivors (score>=50) ordered by edge desc: AAPL(0.10) then SPY(0.02).
    assert [sym for sym, _, _ in out] == ["AAPL", "SPY"]


async def test_run_scan_tick_alerts_top_n_across_all_symbols(
    daemon_context: DaemonContext,
) -> None:
    """top_n is applied across the WHOLE tick, not per symbol: the 2 highest of
    4 floor-passing picks (across 2 symbols) are enqueued."""
    from decimal import Decimal

    from optionsbot.ibkr.types import AccountSummary
    from optionsbot.scoring import ScoredStrategy
    from optionsbot.scoring.types import FactorBreakdown

    def _mk(name: str, score: float) -> ScoredStrategy:
        sug = MagicMock()
        sug.legs = ()
        sug.prob_profit = 0.6
        sug.risk_normalized_expectancy = score / 1000.0
        sug.expected_value = score        # positive -> has_positive_edge True
        # Task 6: new gate requires defined_risk=True and numeric max_loss within cap.
        sug.defined_risk = True
        sug.max_loss = 100.0             # a $100 defined-risk structure fits
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

    # Task 6: patch PositionsClient so the affordability gate sees a large USD
    # equity (all picks have $100 max loss and pass the cap).
    _fake_summary = AccountSummary(
        net_liquidation=Decimal("50000"), buying_power=None,
        available_funds=Decimal("50000"), currency="USD",
    )
    mock_pos = MagicMock()
    mock_pos.get_account_summary = AsyncMock(return_value=_fake_summary)

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), patch(
        "optionsbot.daemon.scan_runner.scan_symbol", new=AsyncMock(side_effect=fake_scan)
    ), patch(
        "optionsbot.daemon.scan_runner.enqueue_alert", new=AsyncMock(return_value=True)
    ) as mock_enqueue, patch(
        "optionsbot.daemon.scan_runner.PositionsClient", return_value=mock_pos
    ):
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


async def test_resolve_scan_symbols_filters_exact_zero_dte_before_screening(
    daemon_context: DaemonContext,
) -> None:
    """Friday-only names must not consume Tuesday/Thursday full-scan slots."""
    from optionsbot.screener.screen import ScreenCandidate

    daemon_context.settings.scan.auto_screen = True
    daemon_context.settings.scan.dte_target = 0
    daemon_context.settings.scan.dte_window_min = 0
    daemon_context.settings.scan.dte_window_max = 0
    daemon_context.settings.screener.universe = ["SPY", "NVDA", "TSLA"]
    screened = AsyncMock(
        return_value=(
            ScreenCandidate(symbol="SPY", hv_rank=0.5, dollar_volume=1e9),
        )
    )

    with patch(
        "optionsbot.daemon.scan_runner.nyse_session_date",
        return_value=date(2026, 7, 23),
    ), patch(
        "optionsbot.daemon.scan_runner.is_last_nyse_session_of_week",
        return_value=False,
    ), patch(
        "optionsbot.daemon.scan_runner.screen_universe",
        new=screened,
    ):
        resolved = await _resolve_scan_symbols(daemon_context)

    assert resolved == [("SPY", None)]
    assert screened.await_args.args[1] == ("SPY",)


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
    """Each symbol runs under the lock so quote sets cannot interleave."""
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


async def test_run_scan_tick_yields_ibkr_lock_between_symbols(
    daemon_context: DaemonContext,
) -> None:
    """A queued protective exit can acquire market data between symbols."""
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=datetime.now(UTC)))
        conn.execute(insert(watchlist).values(symbol="MSFT", added_at=datetime.now(UTC)))

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    exit_acquired = asyncio.Event()
    calls = 0

    async def fake_scan(symbol, *a, **kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            await release_first.wait()
        else:
            assert exit_acquired.is_set()
        return _fake_scan_result(symbol)

    async def queued_exit() -> None:
        await first_started.wait()
        async with daemon_context.ibkr_lock:
            exit_acquired.set()

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), \
         patch("optionsbot.daemon.scan_runner.scan_symbol", new=AsyncMock(side_effect=fake_scan)):
        scan_task = asyncio.create_task(run_scan_tick(daemon_context))
        exit_task = asyncio.create_task(queued_exit())
        await first_started.wait()
        await asyncio.sleep(0)
        release_first.set()
        await asyncio.gather(scan_task, exit_task)

    assert exit_acquired.is_set()


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


def test_rank_alert_candidates_suppresses_all_negative_edge() -> None:
    from optionsbot.daemon.scan_runner import rank_alert_candidates

    def _pick(sym, score, ev, max_loss):
        scored = MagicMock(score=score)
        scored.suggestion.expected_value = ev
        scored.suggestion.risk_normalized_expectancy = ev / max_loss
        # Task 6: new gate requires defined_risk and max_loss; set both so the
        # test exercises the negative-edge filter, not the affordability gate.
        scored.suggestion.defined_risk = True
        scored.suggestion.max_loss = max_loss
        return (sym, scored, 1)

    # Above floor, but every pick is negative-EV -> nothing alert-worthy.
    picks = [
        _pick("NVDA", 80.0, -83.0, 19397.0),
        _pick("NVDA", 75.0, -49.0, 737.0),
    ]
    assert rank_alert_candidates(picks, score_floor=50.0, account_value_usd=50000.0) == []


def test_rank_alert_candidates_keeps_only_positive_edge() -> None:
    from optionsbot.daemon.scan_runner import rank_alert_candidates

    def _pick(sym, score, ev, max_loss):
        scored = MagicMock(score=score)
        scored.suggestion.expected_value = ev
        scored.suggestion.risk_normalized_expectancy = ev / max_loss
        # Task 6: new gate requires defined_risk and max_loss within cap.
        scored.suggestion.defined_risk = True
        scored.suggestion.max_loss = max_loss
        return (sym, scored, 1)

    pos = _pick("AAPL", 80.0, 12.0, 600.0)      # +EV -> kept
    neg = _pick("NVDA", 85.0, -49.0, 737.0)     # -EV -> dropped despite higher score
    out = rank_alert_candidates([neg, pos], score_floor=50.0, account_value_usd=50000.0)
    assert [sym for sym, _, _ in out] == ["AAPL"]


def test_rank_alert_candidates_uses_single_trade_cap() -> None:
    from optionsbot.daemon.scan_runner import rank_alert_candidates

    def _pick(sym, score, ev, max_loss, defined=True):
        scored = MagicMock(score=score)
        scored.suggestion.expected_value = ev
        scored.suggestion.risk_normalized_expectancy = ev / max_loss if max_loss else 0.0
        scored.suggestion.max_loss = max_loss
        scored.suggestion.defined_risk = defined
        return (sym, scored, 1)

    # equity 5000, single cap 0.15 -> $750 per-trade ceiling.
    fits = _pick("SPY", 80.0, 10.0, 700.0)     # 700 <= 750
    too_big = _pick("TSLA", 90.0, 50.0, 900.0)  # 900 > 750 -> dropped
    out = rank_alert_candidates(
        [too_big, fits], score_floor=50.0,
        account_value_usd=5000.0, single_trade_cap_pct=0.15,
    )
    assert [sym for sym, _, _ in out] == ["SPY"]


def test_candidate_admission_blockers_explain_edge_and_risk_failures() -> None:
    from optionsbot.daemon.scan_runner import candidate_admission_blockers

    scored = MagicMock(score=48.0)
    scored.suggestion.expected_value = -12.5
    scored.suggestion.risk_normalized_expectancy = -0.025
    scored.suggestion.defined_risk = True
    scored.suggestion.max_loss = 900.0

    assert candidate_admission_blockers(
        scored,
        score_floor=50.0,
        account_value_usd=5_000.0,
        single_trade_cap_pct=0.10,
    ) == [
        "score_below_floor(score=48.00,floor=50.00)",
        "non_positive_edge(expected_value=-12.50)",
        "single_contract_risk_over_cap(max_loss=900.00,cap=500.00,cap_pct=0.1000)",
    ]


def test_rank_alert_candidates_fail_closed_without_equity() -> None:
    from optionsbot.daemon.scan_runner import rank_alert_candidates

    def _pick(sym, max_loss):
        scored = MagicMock(score=80.0)
        scored.suggestion.expected_value = 10.0
        scored.suggestion.risk_normalized_expectancy = 0.1
        scored.suggestion.max_loss = max_loss
        scored.suggestion.defined_risk = True
        return (sym, scored, 1)

    out = rank_alert_candidates([_pick("SPY", 400.0)], score_floor=50.0, account_value_usd=None)
    assert out == []  # fail-closed: no equity -> nothing surfaced


def test_rank_alert_candidates_drops_undefined_risk() -> None:
    from optionsbot.daemon.scan_runner import rank_alert_candidates

    scored = MagicMock(score=80.0)
    scored.suggestion.expected_value = 10.0
    scored.suggestion.risk_normalized_expectancy = 0.1
    scored.suggestion.max_loss = None
    scored.suggestion.defined_risk = False
    out = rank_alert_candidates(
        [("QQQ", scored, 1)], score_floor=50.0,
        account_value_usd=5000.0, single_trade_cap_pct=0.15,
    )
    assert out == []  # undefined risk no longer surfaced


async def test_run_scan_tick_logs_no_edge_suppression(
    daemon_context: DaemonContext,
) -> None:
    """A tick whose floor-passing picks all lack positive edge logs the
    suppression, so a silent no-alert tick is distinguishable from 'nothing
    scored above the floor'."""
    from decimal import Decimal

    from optionsbot.ibkr.types import AccountSummary
    from optionsbot.scoring import ScoredStrategy
    from optionsbot.scoring.types import FactorBreakdown

    def _mk(name: str, score: float, ev: float, max_loss: float) -> ScoredStrategy:
        sug = MagicMock()
        sug.legs = ()
        sug.prob_profit = 0.6
        sug.expected_value = ev
        sug.risk_normalized_expectancy = ev / max_loss
        # Task 6: new gate requires defined_risk and numeric max_loss so picks
        # pass the affordability check and are dropped by the edge filter only.
        sug.defined_risk = True
        sug.max_loss = max_loss
        return ScoredStrategy(
            strategy_name=name, score=score,
            factors=FactorBreakdown(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            suggestion=sug, rationale="...",
        )

    daemon_context.settings.scan.score_threshold = 50
    # Both pass the score floor; both are negative-EV -> suppressed by edge filter.
    scored = (_mk("csp", 80.0, -83.0, 19397.0), _mk("spread", 75.0, -49.0, 737.0))
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="NVDA", added_at=datetime.now(UTC)))

    # Task 6: supply a large USD equity so picks pass the single-trade cap
    # (cap = 0.10 * 50000 = 5000; max_loss is 19397/737 which exceeds it for
    # the csp but spread is 737 <= 5000). The no-edge path fires when ALL
    # candidates that passed affordability also fail the edge filter.
    # To keep the test about NO-EDGE suppression (not affordability), use a very
    # large account so ALL picks pass the cap and are only dropped by edge filter.
    _fake_summary = AccountSummary(
        net_liquidation=Decimal("500000"), buying_power=None,
        available_funds=Decimal("500000"), currency="USD",
    )
    mock_pos = MagicMock()
    mock_pos.get_account_summary = AsyncMock(return_value=_fake_summary)

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), \
         patch("optionsbot.daemon.scan_runner.scan_symbol",
               new=AsyncMock(return_value=_scan_result_for("NVDA", scored))), \
         patch("optionsbot.daemon.scan_runner.enqueue_alert", new=AsyncMock()) as mock_enq, \
         patch("optionsbot.daemon.scan_runner.PositionsClient", return_value=mock_pos), \
         patch("optionsbot.daemon.scan_runner.log") as mock_log:
        summary = await run_scan_tick(daemon_context)

    mock_enq.assert_not_awaited()                # all -EV -> nothing enqueued
    assert summary.alerts_enqueued == 0
    logged = " ".join(str(call.args[0]) for call in mock_log.info.call_args_list)
    assert "no-edge" in logged


async def test_run_scan_tick_prunes_expired_contracts_from_resolver_cache(
    daemon_context: DaemonContext,
) -> None:
    """Each tick evicts expired OPT entries from the shared resolver cache while
    leaving live ones, so the long-lived cache stays bounded across trading days
    (IBK-148). Far past/future expiries keep this independent of the run date."""
    from optionsbot.ibkr.contracts import _contract_cache_key

    stale = _contract_cache_key("OPT", "SPY", "20000101", 400.0, "C")  # long expired
    live = _contract_cache_key("OPT", "SPY", "20991231", 400.0, "C")   # far future
    daemon_context.resolver._cache[stale] = MagicMock()
    daemon_context.resolver._cache[live] = MagicMock()

    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=datetime.now(UTC)))

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), \
         patch(
            "optionsbot.daemon.scan_runner.scan_symbol",
            new=AsyncMock(side_effect=lambda s, *a, **kw: _fake_scan_result(s)),
         ):
        await run_scan_tick(daemon_context)

    assert stale not in daemon_context.resolver._cache  # expired -> pruned
    assert live in daemon_context.resolver._cache        # live -> kept


async def test_run_scan_tick_skips_symbol_that_exceeds_budget(
    daemon_context: DaemonContext,
) -> None:
    """A symbol whose scan exceeds the per-symbol budget is timed out + skipped;
    the rest of the watchlist still scans (IBK-149)."""
    daemon_context.settings.scan.scan_symbol_timeout_s = 0.2
    with daemon_context.engine.begin() as conn:
        for s in ("AAPL", "SLOW", "MSFT"):
            conn.execute(insert(watchlist).values(symbol=s, added_at=datetime.now(UTC)))

    async def fake_scan(symbol, *a, **kw):
        if symbol == "SLOW":
            await asyncio.sleep(5)  # exceeds the 0.2s budget -> timed out
        return _fake_scan_result(symbol)

    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), \
         patch("optionsbot.daemon.scan_runner.scan_symbol", new=AsyncMock(side_effect=fake_scan)):
        summary = await run_scan_tick(daemon_context)

    assert summary.tickers_scanned == 2  # AAPL + MSFT; SLOW skipped
    assert any("SLOW" in e for e in summary.errors)


async def test_resolve_scan_symbols_falls_back_when_screen_exceeds_budget(
    daemon_context: DaemonContext,
) -> None:
    """A universe screen that exceeds the screener budget times out FAST -> the
    tick falls back to watchlist-only rather than waiting it out (IBK-149).

    The screen returns a VALID (empty) result after a long sleep, so the only
    thing that makes this finish quickly is the timeout firing -- the timing
    assert gates the new budget (without it, the call would wait the full 5s)."""
    daemon_context.settings.scan.auto_screen = True
    daemon_context.settings.scan.screen_timeout_s = 0.2
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=datetime.now(UTC)))

    async def slow_screen(*a, **kw):
        await asyncio.sleep(5)
        return ()  # valid empty result -> WITHOUT the budget this awaits the full 5s

    start = time.monotonic()
    with patch(
        "optionsbot.daemon.scan_runner.screen_universe",
        new=AsyncMock(side_effect=slow_screen),
    ):
        resolved = await _resolve_scan_symbols(daemon_context)
    elapsed = time.monotonic() - start

    assert resolved == [("AAPL", None)]  # timed out -> watchlist-only
    assert elapsed < 2.0  # the 0.2s budget fired; the tick did NOT wait the 5s


async def test_run_scan_tick_bounds_hung_account_summary(
    daemon_context: DaemonContext,
) -> None:
    """The end-of-tick net-liq fetch (an IBKR await held under ibkr_lock) is
    bounded, so a Gateway that wedges after the symbol loop can't hang the tick
    and starve orders management; the tick completes with affordability off
    (IBK-149)."""
    daemon_context.settings.scan.scan_symbol_timeout_s = 0.2
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=datetime.now(UTC)))

    async def hang_summary() -> object:
        await asyncio.sleep(5)
        return MagicMock()

    mock_pos = MagicMock()
    mock_pos.get_account_summary = AsyncMock(side_effect=hang_summary)

    start = time.monotonic()
    with patch("optionsbot.daemon.scan_runner.is_market_open", return_value=True), \
         patch(
            "optionsbot.daemon.scan_runner.scan_symbol",
            new=AsyncMock(side_effect=lambda s, *a, **kw: _fake_scan_result(s)),
         ), \
         patch("optionsbot.daemon.scan_runner.PositionsClient", return_value=mock_pos):
        summary = await run_scan_tick(daemon_context)
    elapsed = time.monotonic() - start

    assert summary.tickers_scanned == 1  # tick completed despite the hung net-liq fetch
    assert elapsed < 2.0  # net-liq timeout fired; did NOT wait the 5s under the lock
