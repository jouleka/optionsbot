"""Deterministic, shadow-only quality features for opening-range setups.

The values in this module are measurements, not an entry score.  They are
persisted so a later walk-forward calibration can determine which features are
useful without letting an unvalidated heuristic change order admission today.
Every bounded transform is monotonic and documented; there are no learned
weights or trading thresholds here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, cast

import pandas as pd

from optionsbot.analysis.types import MarketView

SignalDirection = Literal["bull", "bear"]
SetupType = Literal["fvg_retest", "range_level_retest"]

_ATR_WINDOW = 14
_QUALITY_SCHEMA = "opening_range_quality_v1"


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _positive_ratio_to_unit(value: float | None) -> float | None:
    """Map a non-negative ratio monotonically onto [0, 1)."""
    if value is None or not math.isfinite(value) or value < 0:
        return None
    return value / (1.0 + value)


@dataclass(frozen=True, slots=True)
class OpeningRangeShapeFeatures:
    width: float
    midpoint: float
    width_pct: float | None
    atr_14: float | None
    width_atr_ratio: float | None
    width_atr_normalized: float | None


@dataclass(frozen=True, slots=True)
class BreakoutQualityFeatures:
    displacement: float
    displacement_or_ratio: float
    displacement_normalized: float
    candle_range: float
    body_fraction: float
    directional_body_fraction: float
    rejection_wick_fraction: float
    directional_close_location: float
    volume: float | None
    relative_volume: float | None
    relative_volume_normalized: float | None


@dataclass(frozen=True, slots=True)
class GapQualityFeatures:
    size: float
    size_or_ratio: float
    size_atr_ratio: float | None
    size_normalized: float
    formation_lag_bars: float


@dataclass(frozen=True, slots=True)
class RetestQualityFeatures:
    depth_fraction: float
    rejection_fraction: float
    body_fraction: float
    directional_body_fraction: float
    directional_close_location: float
    volume: float | None
    relative_volume: float | None
    relative_volume_normalized: float | None
    lag_bars: float


@dataclass(frozen=True, slots=True)
class VWAPQualityFeatures:
    value: float | None
    directional_distance_pct: float | None
    directional_distance_or_ratio: float | None
    directional_distance_normalized: float | None
    direction_aligned: bool | None


@dataclass(frozen=True, slots=True)
class TimingQualityFeatures:
    breakout_minutes_from_open: float
    breakout_time_fraction: float
    confirmation_minutes_from_open: float
    confirmation_time_fraction: float
    confirmation_age_minutes: float
    confirmation_age_bars: float
    freshness_normalized: float
    entry_window_remaining_fraction: float


@dataclass(frozen=True, slots=True)
class OpeningRangeQualityFeatures:
    """Feature-only representation of one confirmed OR/FVG setup.

    ``admission_enabled`` is intentionally fixed to ``False``.  Consumers may
    record or display these values, but execution must not infer a pass/fail
    decision from them until a separately versioned model is calibrated.
    """

    setup_type: SetupType
    direction: SignalDirection
    opening_range: OpeningRangeShapeFeatures
    breakout: BreakoutQualityFeatures
    gap: GapQualityFeatures
    retest: RetestQualityFeatures
    vwap: VWAPQualityFeatures
    timing: TimingQualityFeatures
    schema_version: str = _QUALITY_SCHEMA
    calibration_status: str = "shadow_unvalidated"
    admission_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _atr_14(frame: pd.DataFrame) -> float | None:
    """Return a conventional Wilder ATR(14), or ``None`` while unavailable."""
    if len(frame) < _ATR_WINDOW:
        return None
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(
        alpha=1.0 / _ATR_WINDOW,
        adjust=False,
        min_periods=_ATR_WINDOW,
    ).mean()
    if atr.empty:
        return None
    return _finite(atr.iloc[-1])


def _bar(frame: pd.DataFrame, timestamp: datetime) -> pd.Series:
    row = frame.loc[pd.Timestamp(timestamp)]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def _volume(row: pd.Series) -> float | None:
    if "volume" not in row:
        return None
    value = _finite(row["volume"])
    return value if value is not None and value >= 0 else None


def _relative_volume(
    frame: pd.DataFrame,
    *,
    timestamp: datetime,
    current_volume: float | None,
) -> float | None:
    if current_volume is None or "volume" not in frame:
        return None
    previous = pd.to_numeric(
        frame.loc[frame.index < pd.Timestamp(timestamp), "volume"],
        errors="coerce",
    )
    previous = previous[(previous > 0) & previous.map(math.isfinite)]
    if previous.empty:
        return None
    baseline = _finite(previous.mean())
    if baseline is None or baseline <= 0:
        return None
    return current_volume / baseline


def _session_vwap(frame: pd.DataFrame) -> float | None:
    if "volume" not in frame or frame.empty:
        return None
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    typical = (high + low + close) / 3.0
    usable = volume.notna() & typical.notna() & (volume > 0)
    if not usable.any():
        return None
    denominator = _finite(volume[usable].sum())
    numerator = _finite((typical[usable] * volume[usable]).sum())
    if denominator is None or numerator is None or denominator <= 0:
        return None
    return numerator / denominator


def build_opening_range_quality_features(
    frame: pd.DataFrame,
    *,
    direction: SignalDirection,
    setup_type: SetupType,
    market_open: datetime,
    entry_window_minutes: int,
    timeframe_minutes: int,
    now: datetime,
    opening_range_high: float,
    opening_range_low: float,
    breakout_ts: datetime,
    formed_ts: datetime,
    respected_ts: datetime,
    gap_low: float,
    gap_high: float,
) -> OpeningRangeQualityFeatures:
    """Extract reproducible, point-in-time features for a confirmed setup.

    Only bars available through the completed retest participate in ATR, VWAP,
    or relative-volume calculations.  This prevents future bars from leaking
    into the stored feature vector when an older signal is inspected later.
    """
    if timeframe_minutes <= 0 or entry_window_minutes <= 0:
        raise ValueError("timeframe and entry window must be positive")
    range_size = opening_range_high - opening_range_low
    if not range_size > 0:
        raise ValueError("opening range must have positive width")

    through_retest = frame.loc[frame.index <= pd.Timestamp(respected_ts)].copy()
    breakout_bar = _bar(through_retest, breakout_ts)
    retest_bar = _bar(through_retest, respected_ts)
    atr = _atr_14(through_retest)

    range_midpoint = (opening_range_high + opening_range_low) / 2.0
    range_width_pct = range_size / range_midpoint if range_midpoint > 0 else None
    range_atr_ratio = range_size / atr if atr is not None and atr > 0 else None

    breakout_open = float(breakout_bar["open"])
    breakout_high = float(breakout_bar["high"])
    breakout_low = float(breakout_bar["low"])
    breakout_close = float(breakout_bar["close"])
    breakout_range = max(0.0, breakout_high - breakout_low)
    boundary = opening_range_high if direction == "bull" else opening_range_low
    displacement = max(
        0.0,
        breakout_close - boundary if direction == "bull" else boundary - breakout_close,
    )
    displacement_ratio = displacement / range_size
    breakout_body = abs(breakout_close - breakout_open)
    directional_body = max(
        0.0,
        breakout_close - breakout_open
        if direction == "bull"
        else breakout_open - breakout_close,
    )
    if breakout_range > 0:
        body_fraction = _unit(breakout_body / breakout_range)
        directional_body_fraction = _unit(directional_body / breakout_range)
        rejection_wick = (
            breakout_high - max(breakout_open, breakout_close)
            if direction == "bull"
            else min(breakout_open, breakout_close) - breakout_low
        )
        rejection_wick_fraction = _unit(max(0.0, rejection_wick) / breakout_range)
        directional_close_location = _unit(
            (breakout_close - breakout_low) / breakout_range
            if direction == "bull"
            else (breakout_high - breakout_close) / breakout_range
        )
    else:
        body_fraction = 0.0
        directional_body_fraction = 0.0
        rejection_wick_fraction = 0.0
        directional_close_location = 0.0

    breakout_volume = _volume(breakout_bar)
    breakout_rvol = _relative_volume(
        through_retest,
        timestamp=breakout_ts,
        current_volume=breakout_volume,
    )

    gap_size = max(0.0, gap_high - gap_low)
    gap_or_ratio = gap_size / range_size
    gap_atr_ratio = gap_size / atr if atr is not None and atr > 0 else None

    retest_open = float(retest_bar["open"])
    retest_high = float(retest_bar["high"])
    retest_low = float(retest_bar["low"])
    retest_close = float(retest_bar["close"])
    retest_range = max(0.0, retest_high - retest_low)
    if setup_type == "fvg_retest" and gap_size > 0:
        depth = (
            (gap_high - retest_low) / gap_size
            if direction == "bull"
            else (retest_high - gap_low) / gap_size
        )
        rejection_distance = (
            retest_close - gap_high
            if direction == "bull"
            else gap_low - retest_close
        )
    else:
        # Range-level retests permit at most 10% OR penetration in the detector.
        allowed_penetration = 0.10 * range_size
        depth = (
            (boundary - retest_low) / allowed_penetration
            if direction == "bull"
            else (retest_high - boundary) / allowed_penetration
        )
        rejection_distance = (
            retest_close - boundary
            if direction == "bull"
            else boundary - retest_close
        )
    depth_fraction = _unit(depth)
    rejection_fraction = (
        _unit(max(0.0, rejection_distance) / retest_range)
        if retest_range > 0
        else 0.0
    )
    retest_body = abs(retest_close - retest_open)
    retest_directional_body = max(
        0.0,
        retest_close - retest_open
        if direction == "bull"
        else retest_open - retest_close,
    )
    if retest_range > 0:
        retest_body_fraction = _unit(retest_body / retest_range)
        retest_directional_body_fraction = _unit(
            retest_directional_body / retest_range
        )
        retest_close_location = _unit(
            (retest_close - retest_low) / retest_range
            if direction == "bull"
            else (retest_high - retest_close) / retest_range
        )
    else:
        retest_body_fraction = 0.0
        retest_directional_body_fraction = 0.0
        retest_close_location = 0.0

    retest_volume = _volume(retest_bar)
    retest_rvol = _relative_volume(
        through_retest,
        timestamp=respected_ts,
        current_volume=retest_volume,
    )

    vwap = _session_vwap(through_retest)
    directional_vwap_distance_pct: float | None = None
    directional_vwap_distance_or: float | None = None
    directional_vwap_distance_normalized: float | None = None
    vwap_aligned: bool | None = None
    if vwap is not None and vwap > 0:
        signed_distance = (
            retest_close - vwap if direction == "bull" else vwap - retest_close
        )
        directional_vwap_distance_pct = signed_distance / vwap
        directional_vwap_distance_or = signed_distance / range_size
        directional_vwap_distance_normalized = math.tanh(
            directional_vwap_distance_or
        )
        vwap_aligned = signed_distance >= 0

    breakout_minutes = (breakout_ts - market_open).total_seconds() / 60.0
    confirmation_complete = respected_ts + timedelta(minutes=timeframe_minutes)
    confirmation_minutes = (confirmation_complete - market_open).total_seconds() / 60.0
    age_minutes = max(0.0, (now - confirmation_complete).total_seconds() / 60.0)
    age_bars = age_minutes / timeframe_minutes
    entry_end = market_open + timedelta(minutes=entry_window_minutes)
    remaining_minutes = max(0.0, (entry_end - confirmation_complete).total_seconds() / 60.0)

    return OpeningRangeQualityFeatures(
        setup_type=setup_type,
        direction=direction,
        opening_range=OpeningRangeShapeFeatures(
            width=range_size,
            midpoint=range_midpoint,
            width_pct=range_width_pct,
            atr_14=atr,
            width_atr_ratio=range_atr_ratio,
            width_atr_normalized=_positive_ratio_to_unit(range_atr_ratio),
        ),
        breakout=BreakoutQualityFeatures(
            displacement=displacement,
            displacement_or_ratio=displacement_ratio,
            displacement_normalized=cast(
                float, _positive_ratio_to_unit(displacement_ratio)
            ),
            candle_range=breakout_range,
            body_fraction=body_fraction,
            directional_body_fraction=directional_body_fraction,
            rejection_wick_fraction=rejection_wick_fraction,
            directional_close_location=directional_close_location,
            volume=breakout_volume,
            relative_volume=breakout_rvol,
            relative_volume_normalized=_positive_ratio_to_unit(breakout_rvol),
        ),
        gap=GapQualityFeatures(
            size=gap_size,
            size_or_ratio=gap_or_ratio,
            size_atr_ratio=gap_atr_ratio,
            size_normalized=cast(float, _positive_ratio_to_unit(gap_or_ratio)),
            formation_lag_bars=max(
                0.0,
                (formed_ts - breakout_ts).total_seconds()
                / (60.0 * timeframe_minutes),
            ),
        ),
        retest=RetestQualityFeatures(
            depth_fraction=depth_fraction,
            rejection_fraction=rejection_fraction,
            body_fraction=retest_body_fraction,
            directional_body_fraction=retest_directional_body_fraction,
            directional_close_location=retest_close_location,
            volume=retest_volume,
            relative_volume=retest_rvol,
            relative_volume_normalized=_positive_ratio_to_unit(retest_rvol),
            lag_bars=max(
                0.0,
                (respected_ts - formed_ts).total_seconds()
                / (60.0 * timeframe_minutes),
            ),
        ),
        vwap=VWAPQualityFeatures(
            value=vwap,
            directional_distance_pct=directional_vwap_distance_pct,
            directional_distance_or_ratio=directional_vwap_distance_or,
            directional_distance_normalized=directional_vwap_distance_normalized,
            direction_aligned=vwap_aligned,
        ),
        timing=TimingQualityFeatures(
            breakout_minutes_from_open=breakout_minutes,
            breakout_time_fraction=_unit(breakout_minutes / entry_window_minutes),
            confirmation_minutes_from_open=confirmation_minutes,
            confirmation_time_fraction=_unit(
                confirmation_minutes / entry_window_minutes
            ),
            confirmation_age_minutes=age_minutes,
            confirmation_age_bars=age_bars,
            freshness_normalized=1.0 / (1.0 + age_bars),
            entry_window_remaining_fraction=_unit(
                remaining_minutes / entry_window_minutes
            ),
        ),
    )


def quality_payload_with_regime(
    features: OpeningRangeQualityFeatures,
    raw_view: MarketView,
) -> dict[str, Any]:
    """Expose raw inferred regime flags beside the shadow feature vector."""
    payload = features.to_dict()
    opposed = raw_view.direction not in {"neutral", features.direction}
    payload["regime"] = {
        "raw_direction": raw_view.direction,
        "raw_direction_strength": raw_view.direction_strength,
        "raw_iv_regime": raw_view.iv_regime,
        "direction_aligned": raw_view.direction == features.direction,
        "direction_opposed": opposed,
        "direction_neutral": raw_view.direction == "neutral",
        "strong_trend": raw_view.direction_strength == "strong",
        "iv_low": raw_view.iv_regime == "low",
        "iv_neutral": raw_view.iv_regime == "neutral",
        "iv_high": raw_view.iv_regime == "high",
        "earnings_in_window": raw_view.earnings_in_window,
        "iv_warming_up": raw_view.warming_up,
        "iv_rank_is_proxy": raw_view.iv_rank_is_proxy,
    }
    return payload
