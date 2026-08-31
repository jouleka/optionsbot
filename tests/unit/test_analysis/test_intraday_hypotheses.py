"""Causal, shadow-only intraday hypothesis generator tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from optionsbot.analysis.intraday_hypotheses import (
    HYPOTHESIS_GENERATOR_REGISTRY,
    FailedBreakoutFeatures,
    HypothesisResearchConfig,
    LateSessionMomentumFeatures,
    OpeningMomentumFeatures,
    generate_failed_breakout_reversals,
    generate_late_session_momentum,
    generate_opening_momentum_continuation,
    generate_shadow_hypotheses,
)

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
SESSION = datetime(2026, 8, 28, 9, 30, tzinfo=NY)


def _frame(
    rows: list[tuple[float, float, float, float, float]],
    *,
    start: datetime = SESSION,
) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["open", "high", "low", "close", "volume"],
        index=[
            (start + timedelta(minutes=offset)).astimezone(UTC)
            for offset in range(len(rows))
        ],
    )


def _opening_momentum_rows(*, bull: bool = True) -> list[tuple[float, float, float, float, float]]:
    rows: list[tuple[float, float, float, float, float]] = []
    sign = 1.0 if bull else -1.0
    for offset in range(30):
        open_price = 100.0 + sign * offset * 0.02
        close_price = open_price + sign * 0.02
        rows.append(
            (
                open_price,
                max(open_price, close_price) + 0.03,
                min(open_price, close_price) - 0.03,
                close_price,
                100.0 if offset < 15 else 200.0,
            )
        )
    return rows


def _opening_range() -> list[tuple[float, float, float, float, float]]:
    return [(99.5, 100.0, 99.0, 99.5, 100.0) for _ in range(10)]


def _late_session_rows(*, bull: bool = True) -> list[tuple[float, float, float, float, float]]:
    rows: list[tuple[float, float, float, float, float]] = []
    sign = 1.0 if bull else -1.0
    for offset in range(330):  # 09:30 through the 14:59 bar
        if offset < 300:
            open_price = close_price = 100.0
            volume = 100.0
        else:
            late_offset = offset - 300
            open_price = 100.0 + sign * late_offset * 0.02
            close_price = open_price + sign * 0.02
            volume = 200.0
        rows.append(
            (
                open_price,
                max(open_price, close_price) + 0.03,
                min(open_price, close_price) - 0.03,
                close_price,
                volume,
            )
        )
    return rows


@pytest.mark.parametrize(
    ("bull", "direction"),
    [(True, "bull"), (False, "bear")],
)
def test_opening_momentum_is_typed_timestamped_and_shadow_only(
    bull: bool,
    direction: str,
) -> None:
    bars = _frame(_opening_momentum_rows(bull=bull))

    hypotheses = generate_opening_momentum_continuation(
        bars,
        symbol="spy",
        observed_at=datetime(2026, 8, 28, 10, 1, tzinfo=NY),
    )

    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert hypothesis.generator == "opening_momentum_continuation"
    assert hypothesis.direction == direction
    assert hypothesis.symbol == "SPY"
    assert hypothesis.session == "2026-08-28"
    assert hypothesis.option_expiry == "20260828"
    assert hypothesis.signal_at == datetime(2026, 8, 28, 14, 0, tzinfo=UTC)
    assert hypothesis.causal_cutoff_at == hypothesis.signal_at
    assert hypothesis.observed_at > hypothesis.causal_cutoff_at
    assert hypothesis.thesis_expires_at == datetime(2026, 8, 28, 15, 30, tzinfo=UTC)
    assert hypothesis.admission_enabled is False
    assert hypothesis.calibration_status == "shadow_unvalidated"
    assert hypothesis.authority == "shadow_research_only_no_order_or_halt_authority"
    assert isinstance(hypothesis.features, OpeningMomentumFeatures)
    assert hypothesis.features.causal_window.bar_count == 30
    assert hypothesis.features.second_half_volume_ratio == pytest.approx(2.0)
    assert hypothesis.features.second_half_volume_normalized == pytest.approx(2.0 / 3.0)
    assert hypothesis.features.momentum.directional_return_pct > 0.0
    assert 0.0 <= hypothesis.features.momentum.directional_efficiency <= 1.0
    assert 0.0 <= hypothesis.features.momentum.directional_close_location <= 1.0
    assert json.loads(json.dumps(hypothesis.to_dict()))["option_expiry"] == "20260828"


def test_opening_momentum_ignores_future_and_in_progress_bars() -> None:
    bars = _frame(_opening_momentum_rows())
    observed_at = datetime(2026, 8, 28, 10, 1, tzinfo=NY)
    baseline = generate_opening_momentum_continuation(
        bars,
        symbol="SPY",
        observed_at=observed_at,
    )
    future = _frame(
        [(500.0, 600.0, 400.0, 450.0, 1_000_000.0)],
        start=datetime(2026, 8, 28, 10, 5, tzinfo=NY),
    )
    mutated = pd.concat([bars, future])

    assert generate_opening_momentum_continuation(
        mutated,
        symbol="SPY",
        observed_at=observed_at,
    ) == baseline
    assert not generate_opening_momentum_continuation(
        bars,
        symbol="SPY",
        observed_at=datetime(2026, 8, 28, 9, 59, 30, tzinfo=NY),
    )
    missing = bars.drop(bars.index[8])
    assert not generate_opening_momentum_continuation(
        missing,
        symbol="SPY",
        observed_at=observed_at,
    )


def test_failed_high_breakout_reentry_generates_bear_reversal() -> None:
    bars = _frame(
        _opening_range()
        + [
            (99.8, 100.8, 99.7, 100.5, 300.0),  # completed close breakout
            (100.5, 100.7, 100.2, 100.4, 200.0),
            (100.4, 100.5, 99.6, 99.8, 250.0),  # close back inside OR
        ]
    )

    hypotheses = generate_failed_breakout_reversals(
        bars,
        symbol="SPY",
        observed_at=datetime(2026, 8, 28, 9, 44, tzinfo=NY),
    )

    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert hypothesis.direction == "bear"
    assert hypothesis.signal_at == datetime(2026, 8, 28, 13, 43, tzinfo=UTC)
    assert hypothesis.thesis_expires_at == datetime(2026, 8, 28, 14, 43, tzinfo=UTC)
    assert hypothesis.invalidation_level == pytest.approx(100.8)
    assert isinstance(hypothesis.features, FailedBreakoutFeatures)
    features = hypothesis.features
    assert features.failed_side == "high"
    assert features.breakout_close_displacement_or_ratio == pytest.approx(0.5)
    assert features.breakout_extreme_excursion_or_ratio == pytest.approx(0.8)
    assert features.breakout_relative_volume == pytest.approx(3.0)
    assert features.bars_to_reentry == 2
    assert features.reentry_depth_fraction == pytest.approx(0.2)
    assert features.causal_window.start_at == SESSION
    assert features.causal_window.last_bar_completed_at == hypothesis.signal_at


def test_failed_low_breakout_reentry_generates_bull_reversal() -> None:
    bars = _frame(
        _opening_range()
        + [
            (99.2, 99.3, 98.2, 98.5, 300.0),
            (98.5, 99.4, 98.4, 99.2, 250.0),
        ]
    )

    hypotheses = generate_failed_breakout_reversals(
        bars,
        symbol="QQQ",
        observed_at=datetime(2026, 8, 28, 9, 43, tzinfo=NY),
    )

    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert hypothesis.direction == "bull"
    assert hypothesis.invalidation_level == pytest.approx(98.2)
    assert isinstance(hypothesis.features, FailedBreakoutFeatures)
    assert hypothesis.features.failed_side == "low"
    assert hypothesis.features.reentry_depth_fraction == pytest.approx(0.2)


def test_failed_breakout_requires_close_and_bounded_reentry() -> None:
    wick_only = _frame(
        _opening_range()
        + [
            (99.7, 100.8, 99.6, 99.9, 300.0),
            (99.9, 100.1, 99.5, 99.8, 200.0),
        ]
    )
    assert not generate_failed_breakout_reversals(
        wick_only,
        symbol="SPY",
        observed_at=datetime(2026, 8, 28, 9, 43, tzinfo=NY),
    )

    late_reentry = _frame(
        _opening_range()
        + [(99.8, 100.8, 99.7, 100.5, 300.0)]
        + [(100.5, 100.7, 100.2, 100.4, 200.0) for _ in range(5)]
        + [(100.4, 100.5, 99.6, 99.8, 250.0)]
    )
    assert not generate_failed_breakout_reversals(
        late_reentry,
        symbol="SPY",
        observed_at=datetime(2026, 8, 28, 9, 50, tzinfo=NY),
    )


@pytest.mark.parametrize(
    ("bull", "direction"),
    [(True, "bull"), (False, "bear")],
)
def test_late_session_momentum_uses_fixed_causal_window_and_close_expiry(
    bull: bool,
    direction: str,
) -> None:
    bars = _frame(_late_session_rows(bull=bull))

    hypotheses = generate_late_session_momentum(
        bars,
        symbol="NVDA",
        observed_at=datetime(2026, 8, 28, 15, 1, tzinfo=NY),
    )

    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]
    assert hypothesis.direction == direction
    assert hypothesis.signal_at == datetime(2026, 8, 28, 19, 0, tzinfo=UTC)
    assert hypothesis.thesis_expires_at == datetime(2026, 8, 28, 20, 0, tzinfo=UTC)
    assert isinstance(hypothesis.features, LateSessionMomentumFeatures)
    features = hypothesis.features
    assert features.causal_window.start_at == datetime(2026, 8, 28, 14, 30, tzinfo=NY)
    assert features.causal_window.bar_count == 30
    assert features.minutes_from_open == 300.0
    assert features.minutes_to_close == 60.0
    assert features.momentum.relative_volume == pytest.approx(2.0)
    assert features.momentum.directional_return_pct > 0.0
    assert hypothesis.admission_enabled is False


def test_late_session_momentum_waits_for_completed_window() -> None:
    bars = _frame(_late_session_rows())
    assert not generate_late_session_momentum(
        bars,
        symbol="SPY",
        observed_at=datetime(2026, 8, 28, 14, 59, 30, tzinfo=NY),
    )


def test_registry_is_complete_selectable_and_deduplicated() -> None:
    assert tuple(HYPOTHESIS_GENERATOR_REGISTRY) == (
        "opening_momentum_continuation",
        "failed_breakout_reversal",
        "late_session_momentum",
    )
    bars = _frame(_opening_momentum_rows())
    observed_at = datetime(2026, 8, 28, 10, 1, tzinfo=NY)
    hypotheses = generate_shadow_hypotheses(
        bars,
        symbol="SPY",
        observed_at=observed_at,
        generators=("opening_momentum_continuation",),
    )
    assert len(hypotheses) == 1
    assert hypotheses[0].generator == "opening_momentum_continuation"
    assert len({item.hypothesis_id for item in hypotheses}) == len(hypotheses)
    with pytest.raises(ValueError, match="unknown shadow hypothesis"):
        generate_shadow_hypotheses(
            bars,
            symbol="SPY",
            observed_at=observed_at,
            generators=("not_registered",),  # type: ignore[arg-type]
        )


def test_identity_includes_research_parameter_version() -> None:
    bars = _frame(_opening_momentum_rows())
    observed_at = datetime(2026, 8, 28, 10, 1, tzinfo=NY)
    first = generate_opening_momentum_continuation(
        bars,
        symbol="SPY",
        observed_at=observed_at,
        config=HypothesisResearchConfig(parameter_version="research-a"),
    )[0]
    second = generate_opening_momentum_continuation(
        bars,
        symbol="SPY",
        observed_at=observed_at,
        config=HypothesisResearchConfig(parameter_version="research-b"),
    )[0]
    assert first.hypothesis_id != second.hypothesis_id


def test_invalid_ohlc_is_removed_instead_of_leaking_nan_features() -> None:
    bars = _frame(_opening_momentum_rows())
    bars.loc[bars.index[12], "close"] = float("nan")
    assert not generate_opening_momentum_continuation(
        bars,
        symbol="SPY",
        observed_at=datetime(2026, 8, 28, 10, 1, tzinfo=NY),
    )


def test_research_config_validation_is_local_and_explicit() -> None:
    with pytest.raises(ValueError, match="positive"):
        HypothesisResearchConfig(failed_breakout_max_reentry_bars=0)
    with pytest.raises(ValueError, match="divisible"):
        HypothesisResearchConfig(timeframe_minutes=5, opening_range_minutes=12)
