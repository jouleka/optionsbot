"""Opening-range breakout/FVG setup tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from optionsbot.analysis.opening_range_fvg import detect_opening_range_fvg

NY = ZoneInfo("America/New_York")


def _frame(extra: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    start = datetime(2026, 7, 31, 9, 30, tzinfo=NY)
    opening = [
        (99.50, 100.00, 99.00, 99.60)
        for _ in range(10)
    ]
    rows = opening + extra
    index = [(start + timedelta(minutes=i)).astimezone(ZoneInfo("UTC")) for i in range(len(rows))]
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)


def test_bull_break_fvg_retest_and_respect_confirms_two_r() -> None:
    bars = _frame(
        [
            (99.80, 100.70, 99.70, 100.50),  # close-confirmed range break
            (100.50, 101.00, 100.45, 100.90),
            (100.90, 101.10, 100.80, 101.00),  # FVG 100.70..100.80
            (100.70, 100.95, 100.70, 100.90),  # enters, holds, closes above
        ]
    )

    signal = detect_opening_range_fvg(
        bars,
        symbol="SPY",
        now=datetime(2026, 7, 31, 9, 45, tzinfo=NY),
    )

    assert signal is not None
    assert signal.direction == "bull"
    assert signal.opening_range_high == 100.0
    assert signal.opening_range_low == 99.0
    assert signal.fvg_low == 100.70
    assert signal.fvg_high == 100.80
    assert signal.target_r == 2.0
    assert signal.stop_pct == 0.15
    assert signal.target_pct == 0.30


def test_wick_outside_range_is_not_a_breakout() -> None:
    bars = _frame(
        [
            (99.80, 100.70, 99.70, 99.90),
            (99.90, 100.20, 99.80, 99.95),
            (99.95, 100.10, 99.85, 99.90),
            (99.90, 100.05, 99.80, 99.95),
        ]
    )

    assert (
        detect_opening_range_fvg(
            bars,
            symbol="SPY",
            now=datetime(2026, 7, 31, 9, 45, tzinfo=NY),
        )
        is None
    )


def test_bear_gap_must_hold_its_far_edge() -> None:
    bars = _frame(
        [
            (99.20, 99.30, 98.70, 98.80),
            (98.80, 98.90, 98.30, 98.40),
            (98.30, 98.50, 98.10, 98.20),  # bearish FVG 98.50..98.70
            (98.60, 98.75, 98.45, 98.55),  # breaches 98.70, invalidates
            (98.55, 98.65, 98.20, 98.30),
        ]
    )

    assert (
        detect_opening_range_fvg(
            bars,
            symbol="QQQ",
            now=datetime(2026, 7, 31, 9, 46, tzinfo=NY),
        )
        is None
    )


def test_in_progress_respect_candle_is_ignored() -> None:
    bars = _frame(
        [
            (99.80, 100.70, 99.70, 100.50),
            (100.50, 101.00, 100.45, 100.90),
            (100.90, 101.10, 100.80, 101.00),
            (100.65, 100.95, 100.65, 100.90),
        ]
    )

    assert (
        detect_opening_range_fvg(
            bars,
            symbol="SPY",
            now=datetime(2026, 7, 31, 9, 43, 30, tzinfo=NY),
        )
        is None
    )


def test_strong_opening_range_level_retest_is_an_independent_setup() -> None:
    bars = _frame(
        [
            (99.80, 100.80, 99.70, 100.60),  # strong bull breakout
            (100.60, 101.00, 100.40, 100.80),
            (100.05, 100.70, 99.95, 100.60),  # retests 100 and rejects it
        ]
    )

    signal = detect_opening_range_fvg(
        bars,
        symbol="SPY",
        now=datetime(2026, 7, 31, 9, 44, tzinfo=NY),
    )

    assert signal is not None
    assert signal.direction == "bull"
    assert signal.setup_type == "range_level_retest"
    assert signal.fvg_low == signal.fvg_high == 100.0
    assert signal.target_r == 2.0


def test_later_valid_reversal_is_not_suppressed_by_first_breakout() -> None:
    bars = _frame(
        [
            (99.80, 100.70, 99.70, 100.50),  # first break is a bull fake-out
            (100.40, 100.50, 99.40, 99.50),
            (99.40, 99.50, 98.70, 98.80),  # later bear breakout
            (98.80, 98.90, 98.30, 98.40),
            (98.30, 98.50, 98.10, 98.20),  # bear FVG 98.50..98.70
            (98.65, 98.68, 98.35, 98.40),  # retest and respect
        ]
    )

    signal = detect_opening_range_fvg(
        bars,
        symbol="QQQ",
        now=datetime(2026, 7, 31, 9, 47, tzinfo=NY),
    )

    assert signal is not None
    assert signal.direction == "bear"
    assert signal.setup_type == "fvg_retest"
