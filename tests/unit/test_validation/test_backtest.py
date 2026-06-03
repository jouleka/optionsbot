"""Tests for the calibration-backtest pure core (IBK-99 Phase A)."""

from __future__ import annotations

import math

from optionsbot.strategies.base import Leg
from optionsbot.validation.backtest import (
    calibrate,
    historical_win_rate,
    horizon_trading_days,
)
from optionsbot.validation.types import BacktestRow


def test_horizon_trading_days_scales_calendar_to_trading() -> None:
    assert horizon_trading_days(365) == 252
    assert horizon_trading_days(45) == 31  # round(45*252/365)
    assert horizon_trading_days(1) == 1     # floored at 1


def _long_call(strike: float) -> tuple[Leg, ...]:
    return (Leg(symbol="X", side="buy", sec_type="OPT", expiry="20260101",
                strike=strike, right="C", quantity=1),)


def test_historical_win_rate_long_call_all_up() -> None:
    # A strictly rising series: every 1-day-forward step is up. A long call
    # struck below entry_spot (entry_spot=100) paid 1.00 debit (cod=-1.00*100? no:
    # credit_or_debit is per-set dollars NOT x100 here -- terminal_pnl adds cod
    # directly). Use cod=-50 (=$50 debit) and strike 90 so it finishes ITM.
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    out = historical_win_rate(
        legs=_long_call(90.0), credit_or_debit=-50.0, entry_spot=100.0,
        closes=closes, dte_days=1,
    )
    assert out is not None
    raw, dedrift, n = out
    # n = len(closes) - horizon(=1) = 5 forward returns.
    assert n == 5
    # All up moves -> s_t = 100*exp(+) > 100 >= 90; intrinsic >= 10 -> *100 = >=1000;
    # +cod(-50) > 0 -> raw win-rate 1.0.
    assert raw == 1.0
    assert 0.0 <= dedrift <= 1.0


def test_historical_win_rate_too_few_samples_returns_none() -> None:
    assert historical_win_rate(
        legs=_long_call(90.0), credit_or_debit=-50.0, entry_spot=100.0,
        closes=[100.0], dte_days=1,
    ) is None


def test_calibrate_buckets_by_predicted_pop() -> None:
    rows = [
        BacktestRow(symbol="A", strategy="bull_call_spread", predicted=0.72,
                    raw=0.80, dedrift=0.70, n=100),
        BacktestRow(symbol="B", strategy="bull_call_spread", predicted=0.74,
                    raw=0.60, dedrift=0.55, n=100),
        BacktestRow(symbol="C", strategy="long_call", predicted=0.25,
                    raw=0.30, dedrift=0.20, n=100),
    ]
    report = calibrate(rows, n_buckets=10)
    # Two rows land in the [0.7,0.8) bucket, one in [0.2,0.3).
    by_lo = {round(b.lo, 1): b for b in report.buckets}
    assert by_lo[0.7].count == 2
    assert math.isclose(by_lo[0.7].mean_pred, 0.73, abs_tol=1e-9)
    assert math.isclose(by_lo[0.7].mean_raw, 0.70, abs_tol=1e-9)
    assert by_lo[0.2].count == 1
    assert report.overall_count == 3
    assert "bull_call_spread" in report.by_strategy
    assert report.by_strategy["bull_call_spread"].count == 2
