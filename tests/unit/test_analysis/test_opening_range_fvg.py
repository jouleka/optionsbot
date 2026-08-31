"""Opening-range breakout/FVG setup tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

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


def _frame_with_volume(
    extra: list[tuple[float, float, float, float, float]],
) -> pd.DataFrame:
    start = datetime(2026, 7, 31, 9, 30, tzinfo=NY)
    opening = [(99.50, 100.00, 99.00, 99.60, 100.0) for _ in range(10)]
    rows = opening + extra
    index = [
        (start + timedelta(minutes=i)).astimezone(ZoneInfo("UTC"))
        for i in range(len(rows))
    ]
    return pd.DataFrame(
        rows,
        columns=["open", "high", "low", "close", "volume"],
        index=index,
    )


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
    assert signal.quality is not None
    assert signal.quality.calibration_status == "shadow_unvalidated"
    assert signal.quality.admission_enabled is False


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


@pytest.mark.parametrize(
    "extra",
    [
        [
            (99.80, 100.80, 99.70, 100.60),  # bull breakout
            (100.50, 100.60, 99.70, 100.20),  # wick >10% back inside
            (100.05, 100.70, 99.95, 100.60),  # later apparent respect
        ],
        [
            (99.20, 99.30, 98.20, 98.40),  # bear breakout
            (98.50, 99.30, 98.40, 98.80),  # wick >10% back inside
            (98.95, 99.05, 98.30, 98.40),  # later apparent respect
        ],
    ],
    ids=["bull", "bear"],
)
def test_material_range_reentry_invalidates_later_level_reclaim(
    extra: list[tuple[float, float, float, float]],
) -> None:
    bars = _frame(extra)

    signal = detect_opening_range_fvg(
        bars,
        symbol="SPY",
        now=datetime(2026, 7, 31, 9, 44, tzinfo=NY),
    )

    assert signal is None


def test_material_range_reentry_cannot_be_revived_by_a_later_fvg() -> None:
    bars = _frame(
        [
            (99.80, 100.80, 99.70, 100.60),  # bull breakout
            (100.50, 100.60, 99.70, 100.20),  # material re-entry wick
            (100.20, 101.10, 100.15, 100.95),
            (100.95, 101.20, 100.70, 101.10),  # later FVG 100.60..100.70
            (100.65, 100.90, 100.65, 100.85),  # apparent FVG respect
        ]
    )

    signal = detect_opening_range_fvg(
        bars,
        symbol="SPY",
        now=datetime(2026, 7, 31, 9, 46, tzinfo=NY),
    )

    assert signal is None


@pytest.mark.parametrize(
    "extra",
    [
        [
            (99.80, 100.80, 99.70, 100.60),  # bull breakout
            (100.05, 100.70, 99.95, 100.60),  # valid bull confirmation
            (100.50, 100.60, 99.70, 100.20),  # later material re-entry
        ],
        [
            (99.20, 99.30, 98.20, 98.40),  # bear breakout
            (98.95, 99.05, 98.30, 98.40),  # valid bear confirmation
            (98.50, 100.30, 98.40, 98.80),  # later material re-entry
        ],
    ],
    ids=["bull", "bear"],
)
def test_material_reentry_invalidates_an_earlier_confirmation(
    extra: list[tuple[float, float, float, float]],
) -> None:
    bars = _frame(extra)

    signal = detect_opening_range_fvg(
        bars,
        symbol="SPY",
        now=datetime(2026, 7, 31, 9, 44, tzinfo=NY),
    )

    assert signal is None


@pytest.mark.parametrize(
    "extra",
    [
        [
            (99.80, 100.80, 99.70, 100.60),  # bull breakout
            (100.05, 100.70, 99.95, 100.60),  # valid bull confirmation
            (99.50, 99.60, 98.20, 98.40),  # fresh opposite bear breakout
        ],
        [
            (99.20, 99.30, 98.20, 98.40),  # bear breakout
            (98.95, 99.05, 98.30, 98.40),  # valid bear confirmation
            (99.50, 100.80, 99.40, 100.60),  # fresh opposite bull breakout
        ],
    ],
    ids=["bull-to-bear", "bear-to-bull"],
)
def test_fresh_opposite_break_invalidates_earlier_confirmation_across_segment(
    extra: list[tuple[float, float, float, float]],
) -> None:
    bars = _frame(extra)

    signal = detect_opening_range_fvg(
        bars,
        symbol="SPY",
        now=datetime(2026, 7, 31, 9, 44, tzinfo=NY),
    )

    assert signal is None


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


def test_hour_late_gap_cannot_be_claimed_by_old_breakout() -> None:
    bars = _frame(
        [
            (99.80, 100.70, 99.70, 100.50),  # breakout
            *[(100.40, 100.60, 100.20, 100.45) for _ in range(45)],
            (100.50, 101.00, 100.45, 100.90),
            (100.90, 101.10, 100.80, 101.00),  # unrelated late FVG
            (100.70, 100.95, 100.70, 100.90),
        ]
    )

    signal = detect_opening_range_fvg(
        bars,
        symbol="SPY",
        now=datetime(2026, 7, 31, 10, 45, tzinfo=NY),
        entry_window_minutes=360,
    )

    assert signal is None


def test_bull_fvg_quality_features_include_shape_volume_vwap_and_time() -> None:
    bars = _frame_with_volume(
        [
            (99.80, 100.70, 99.70, 100.50, 200.0),
            (100.50, 101.00, 100.45, 100.90, 150.0),
            (100.90, 101.10, 100.80, 101.00, 180.0),
            (100.70, 100.95, 100.70, 100.90, 250.0),
        ]
    )

    signal = detect_opening_range_fvg(
        bars,
        symbol="SPY",
        now=datetime(2026, 7, 31, 9, 45, tzinfo=NY),
    )

    assert signal is not None and signal.quality is not None
    quality = signal.quality
    assert quality.schema_version == "opening_range_quality_v1"
    assert quality.setup_type == "fvg_retest"
    assert quality.opening_range.width == pytest.approx(1.0)
    assert quality.opening_range.width_pct == pytest.approx(1.0 / 99.5)
    assert quality.opening_range.atr_14 is not None
    assert quality.opening_range.width_atr_ratio is not None
    assert 0.0 < quality.opening_range.width_atr_normalized < 1.0

    assert quality.breakout.displacement == pytest.approx(0.5)
    assert quality.breakout.displacement_or_ratio == pytest.approx(0.5)
    assert quality.breakout.displacement_normalized == pytest.approx(1.0 / 3.0)
    assert quality.breakout.body_fraction == pytest.approx(0.7)
    assert quality.breakout.directional_body_fraction == pytest.approx(0.7)
    assert quality.breakout.rejection_wick_fraction == pytest.approx(0.2)
    assert quality.breakout.directional_close_location == pytest.approx(0.8)
    assert quality.breakout.volume == 200.0
    assert quality.breakout.relative_volume == pytest.approx(2.0)
    assert quality.breakout.relative_volume_normalized == pytest.approx(2.0 / 3.0)

    assert quality.gap.size == pytest.approx(0.1)
    assert quality.gap.size_or_ratio == pytest.approx(0.1)
    assert quality.gap.size_normalized == pytest.approx(0.1 / 1.1)
    assert quality.gap.formation_lag_bars == 2.0
    assert quality.retest.depth_fraction == pytest.approx(1.0)
    assert quality.retest.rejection_fraction == pytest.approx(0.4)
    assert quality.retest.body_fraction == pytest.approx(0.8)
    assert quality.retest.directional_body_fraction == pytest.approx(0.8)
    assert quality.retest.lag_bars == 1.0

    assert quality.vwap.value is not None
    assert quality.vwap.direction_aligned is True
    assert quality.vwap.directional_distance_normalized is not None
    assert -1.0 <= quality.vwap.directional_distance_normalized <= 1.0
    assert quality.timing.breakout_minutes_from_open == 10.0
    assert quality.timing.confirmation_minutes_from_open == 14.0
    assert quality.timing.confirmation_age_minutes == 1.0
    assert quality.timing.confirmation_age_bars == 1.0
    assert quality.timing.freshness_normalized == pytest.approx(0.5)
    assert quality.timing.entry_window_remaining_fraction == pytest.approx(76.0 / 90.0)

    payload = signal.to_dict()["quality"]
    assert payload["calibration_status"] == "shadow_unvalidated"
    assert payload["admission_enabled"] is False


def test_bear_fvg_quality_features_are_directionally_symmetric() -> None:
    bars = _frame_with_volume(
        [
            (99.20, 99.30, 98.70, 98.80, 200.0),
            (98.80, 98.90, 98.30, 98.40, 150.0),
            (98.30, 98.50, 98.10, 98.20, 180.0),
            (98.62, 98.65, 98.35, 98.40, 250.0),
        ]
    )

    signal = detect_opening_range_fvg(
        bars,
        symbol="QQQ",
        now=datetime(2026, 7, 31, 9, 45, tzinfo=NY),
    )

    assert signal is not None and signal.quality is not None
    assert signal.direction == "bear"
    quality = signal.quality
    assert quality.breakout.displacement == pytest.approx(0.2)
    assert quality.breakout.directional_body_fraction == pytest.approx(2.0 / 3.0)
    assert quality.breakout.directional_close_location == pytest.approx(5.0 / 6.0)
    assert quality.retest.depth_fraction == pytest.approx(0.75)
    assert quality.retest.rejection_fraction == pytest.approx(1.0 / 3.0)
    assert quality.retest.directional_close_location == pytest.approx(5.0 / 6.0)
    assert quality.vwap.direction_aligned is True


def test_quality_features_degrade_to_none_without_volume_or_enough_atr_bars() -> None:
    bars = _frame(
        [
            (99.80, 100.80, 99.70, 100.60),
            (100.60, 101.00, 100.40, 100.80),
            (100.05, 100.70, 99.95, 100.60),
        ]
    )

    signal = detect_opening_range_fvg(
        bars,
        symbol="SPY",
        now=datetime(2026, 7, 31, 9, 44, tzinfo=NY),
    )

    assert signal is not None and signal.quality is not None
    quality = signal.quality
    assert quality.setup_type == "range_level_retest"
    assert quality.opening_range.atr_14 is None
    assert quality.opening_range.width_atr_ratio is None
    assert quality.breakout.volume is None
    assert quality.breakout.relative_volume is None
    assert quality.retest.volume is None
    assert quality.vwap.value is None
    assert quality.vwap.direction_aligned is None
    assert quality.gap.size == 0.0
    assert quality.gap.formation_lag_bars == 0.0
    assert quality.retest.depth_fraction == pytest.approx(0.5)
    assert quality.retest.rejection_fraction == pytest.approx(0.8)
