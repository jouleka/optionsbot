"""Tests for the calibration-backtest pure core (IBK-99 Phase A)."""

from __future__ import annotations

import datetime as _dt
import math

import pytest

from optionsbot.strategies.base import Leg
from optionsbot.validation.backtest import (
    calibrate,
    historical_win_rate,
    horizon_trading_days,
    run_backtest,
)
from optionsbot.validation.types import BacktestRow, PickRecord


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


def _pick(symbol: str, pop: float, strike: float) -> PickRecord:
    return PickRecord(
        symbol=symbol, entry_spot=100.0, entry_date=_dt.date(2026, 1, 1),
        expiry="20260201", dte_days=31,
        legs=(Leg(symbol=symbol, side="buy", sec_type="OPT", expiry="20260201",
                  strike=strike, right="C", quantity=1),),
        credit_or_debit=-50.0, prob_profit=pop, score=60.0, strategy="long_call",
    )


@pytest.mark.asyncio
async def test_run_backtest_uses_injected_fetch_and_builds_report() -> None:
    picks = [_pick("AAA", 0.72, 90.0), _pick("BBB", 0.25, 130.0)]
    calls: list[str] = []

    async def fake_fetch(symbol: str) -> list[float]:
        calls.append(symbol)
        # Gently rising series -> long call struck at 90 wins, at 130 loses.
        return [100.0 + i * 0.5 for i in range(60)]

    report = await run_backtest(picks, fake_fetch)
    assert calls == ["AAA", "BBB"]
    assert report.overall_count == 2
    # Symbol fetch is cached per symbol (each fetched once).


def test_calibrate_bucket_boundaries() -> None:
    """Exact deciles land in the right bucket (0.70 -> [0.7,0.8), 1.0 -> last)."""
    rows = [
        BacktestRow(symbol="A", strategy="s", predicted=0.70, raw=0.7, dedrift=0.7, n=10),
        BacktestRow(symbol="B", strategy="s", predicted=0.30, raw=0.3, dedrift=0.3, n=10),
        BacktestRow(symbol="C", strategy="s", predicted=1.0, raw=1.0, dedrift=1.0, n=10),
        BacktestRow(symbol="D", strategy="s", predicted=0.0, raw=0.0, dedrift=0.0, n=10),
    ]
    report = calibrate(rows, n_buckets=10)
    by_lo = {round(b.lo, 1): b for b in report.buckets}
    assert by_lo[0.7].count == 1  # would be 0 with naive int(0.7*10)==6
    assert by_lo[0.3].count == 1
    assert by_lo[0.9].count == 1  # 1.0 pinned to the last bucket, not dropped
    assert by_lo[0.0].count == 1
    assert report.overall_count == 4


def test_load_pick_records_filters_non_modelable(tmp_path) -> None:
    """load_pick_records keeps single-expiry picks with a prob_profit; skips
    multi-expiry (calendar) and null-prob rows."""
    from datetime import UTC, datetime
    from pathlib import Path

    from alembic.config import Config
    from sqlalchemy import insert

    from alembic import command
    from optionsbot.storage.db import create_engine_for_path
    from optionsbot.storage.schema import snapshots as snaps
    from optionsbot.storage.schema import strategy_scores as scores
    from optionsbot.validation.backtest import load_pick_records

    db_path = tmp_path / "v.db"
    root = Path(__file__).resolve().parents[3]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")
    engine = create_engine_for_path(db_path)

    def _leg(side: str, strike: float, expiry: str) -> dict[str, object]:
        return {"symbol": "SPY", "side": side, "sec_type": "OPT",
                "expiry": expiry, "strike": strike, "right": "C", "quantity": 1}

    with engine.begin() as conn:
        sid = conn.execute(
            insert(snaps).values(
                symbol="SPY", ts=datetime(2026, 1, 1, tzinfo=UTC), spot=100.0
            )
        ).inserted_primary_key[0]
        conn.execute(insert(scores).values(
            snapshot_id=sid, strategy="bull_call_spread", score=60.0,
            legs_json=[_leg("buy", 100.0, "20260201"), _leg("sell", 110.0, "20260201")],
            suggestion_json={"prob_profit": 0.5, "credit_or_debit": -200.0}))
        conn.execute(insert(scores).values(  # multi-expiry -> not modelable
            snapshot_id=sid, strategy="calendar", score=55.0,
            legs_json=[_leg("sell", 100.0, "20260201"), _leg("buy", 100.0, "20260301")],
            suggestion_json={"prob_profit": None, "credit_or_debit": -150.0}))
        conn.execute(insert(scores).values(  # null prob_profit -> skipped
            snapshot_id=sid, strategy="long_call", score=50.0,
            legs_json=[_leg("buy", 100.0, "20260201")],
            suggestion_json={"prob_profit": None, "credit_or_debit": -300.0}))

    picks = load_pick_records(engine)
    assert len(picks) == 1
    assert picks[0].strategy == "bull_call_spread"
    assert picks[0].dte_days == 31
