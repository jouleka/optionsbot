"""Independent, point-in-time 0DTE hypotheses for prospective shadow research.

These generators observe completed RTH bars and describe bounded underlying
theses.  They do not select an option, estimate managed probability/EV, rank a
candidate, or authorize an order.  Time-window parameters live in the local
research config rather than production execution settings so they cannot
silently alter admission or risk behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, time, timedelta
from numbers import Real
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, cast
from zoneinfo import ZoneInfo

import pandas as pd

HypothesisDirection = Literal["bull", "bear"]
HypothesisKind = Literal[
    "opening_momentum_continuation",
    "failed_breakout_reversal",
    "late_session_momentum",
]

_NEW_YORK = ZoneInfo("America/New_York")
_SCHEMA_VERSION: Final = "intraday_shadow_hypothesis_v1"
_AUTHORITY: Final = "shadow_research_only_no_order_or_halt_authority"
_ATR_WINDOW = 14


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _positive_ratio_to_unit(value: float | None) -> float | None:
    if value is None or not math.isfinite(value) or value < 0.0:
        return None
    return value / (1.0 + value)


def _utc_iso(value: datetime) -> str:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat()


def _json_ready(value: object) -> object:
    if isinstance(value, datetime):
        return _utc_iso(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class HypothesisResearchConfig:
    """Versioned shadow-window parameters with no production authority."""

    timeframe_minutes: int = 1
    opening_range_minutes: int = 10
    opening_momentum_window_minutes: int = 30
    opening_momentum_lifetime_minutes: int = 90
    failed_breakout_max_reentry_bars: int = 5
    failed_breakout_lifetime_minutes: int = 60
    late_session_start: time = time(14, 30)
    late_session_window_minutes: int = 30
    parameter_version: str = "intraday_shadow_windows_v1"

    def __post_init__(self) -> None:
        positive = (
            self.timeframe_minutes,
            self.opening_range_minutes,
            self.opening_momentum_window_minutes,
            self.opening_momentum_lifetime_minutes,
            self.failed_breakout_max_reentry_bars,
            self.failed_breakout_lifetime_minutes,
            self.late_session_window_minutes,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("shadow hypothesis window parameters must be positive")
        divisible = (
            self.opening_range_minutes,
            self.opening_momentum_window_minutes,
            self.late_session_window_minutes,
        )
        if any(value % self.timeframe_minutes for value in divisible):
            raise ValueError("research windows must be divisible by timeframe_minutes")
        if self.late_session_start.tzinfo is not None:
            raise ValueError("late_session_start must be a naive New York wall time")
        session_start_minutes = 9 * 60 + 30
        late_start_minutes = (
            self.late_session_start.hour * 60 + self.late_session_start.minute
        )
        if (
            late_start_minutes < session_start_minutes
            or late_start_minutes + self.late_session_window_minutes > 16 * 60
        ):
            raise ValueError("late-session research window must fit inside RTH")


@dataclass(frozen=True, slots=True)
class CausalWindow:
    start_at: datetime
    end_at: datetime
    last_bar_started_at: datetime
    last_bar_completed_at: datetime
    bar_count: int


@dataclass(frozen=True, slots=True)
class MomentumMeasurements:
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    return_pct: float
    directional_return_pct: float
    range_pct: float
    atr_14: float | None
    directional_return_atr_ratio: float | None
    directional_return_atr_normalized: float | None
    directional_efficiency: float
    directional_close_location: float
    vwap: float | None
    directional_vwap_distance_pct: float | None
    vwap_direction_aligned: bool | None
    total_volume: float | None
    mean_volume: float | None
    relative_volume: float | None
    relative_volume_normalized: float | None


@dataclass(frozen=True, slots=True)
class OpeningMomentumFeatures:
    causal_window: CausalWindow
    momentum: MomentumMeasurements
    opening_window_minutes: int
    thesis_lifetime_minutes: int
    second_half_volume_ratio: float | None
    second_half_volume_normalized: float | None
    parameter_version: str


@dataclass(frozen=True, slots=True)
class FailedBreakoutFeatures:
    causal_window: CausalWindow
    opening_range_high: float
    opening_range_low: float
    opening_range_width: float
    opening_range_atr_ratio: float | None
    opening_range_minutes: int
    failed_side: Literal["high", "low"]
    breakout_at: datetime
    breakout_close: float
    breakout_extreme: float
    breakout_close_displacement_or_ratio: float
    breakout_extreme_excursion_or_ratio: float
    breakout_rejection_wick_fraction: float
    breakout_volume: float | None
    breakout_relative_volume: float | None
    breakout_relative_volume_normalized: float | None
    reentry_at: datetime
    reentry_close: float
    bars_to_reentry: int
    max_reentry_bars: int
    thesis_lifetime_minutes: int
    reentry_depth_fraction: float
    reentry_directional_body_fraction: float
    reentry_vwap: float | None
    reentry_vwap_direction_aligned: bool | None
    parameter_version: str


@dataclass(frozen=True, slots=True)
class LateSessionMomentumFeatures:
    causal_window: CausalWindow
    momentum: MomentumMeasurements
    minutes_from_open: float
    minutes_to_close: float
    window_minutes: int
    configured_start_local: str
    parameter_version: str


type HypothesisFeatures = (
    OpeningMomentumFeatures | FailedBreakoutFeatures | LateSessionMomentumFeatures
)


@dataclass(frozen=True, slots=True)
class ShadowIntradayHypothesis:
    """A stable, causal research observation with explicitly zero authority."""

    hypothesis_id: str
    generator: HypothesisKind
    symbol: str
    direction: HypothesisDirection
    session: str
    option_expiry: str
    signal_at: datetime
    observed_at: datetime
    causal_cutoff_at: datetime
    thesis_expires_at: datetime
    reference_price: float
    invalidation_level: float
    features: HypothesisFeatures
    schema_version: Literal["intraday_shadow_hypothesis_v1"] = field(
        default=_SCHEMA_VERSION,
        init=False,
    )
    calibration_status: Literal["shadow_unvalidated"] = field(
        default="shadow_unvalidated",
        init=False,
    )
    admission_enabled: Literal[False] = field(default=False, init=False)
    authority: Literal["shadow_research_only_no_order_or_halt_authority"] = field(
        default=_AUTHORITY,
        init=False,
    )

    def to_dict(self) -> dict[str, Any]:
        payload = _json_ready(asdict(self))
        return cast(dict[str, Any], payload)


@dataclass(frozen=True, slots=True)
class _GenerationContext:
    symbol: str
    observed_at: datetime
    session: str
    option_expiry: str
    market_open: datetime
    market_close: datetime
    timeframe_minutes: int
    frame: pd.DataFrame


def _prepare_context(
    bars: pd.DataFrame,
    *,
    symbol: str,
    observed_at: datetime,
    timeframe_minutes: int,
) -> _GenerationContext | None:
    required = {"open", "high", "low", "close"}
    normalized_symbol = symbol.upper().strip()
    if not normalized_symbol:
        raise ValueError("symbol is required")
    if timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes must be positive")
    if bars.empty or not required.issubset(bars.columns):
        return None
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    observed_at = observed_at.astimezone(UTC)
    observed_ny = observed_at.astimezone(_NEW_YORK)
    market_open = datetime.combine(observed_ny.date(), time(9, 30), tzinfo=_NEW_YORK)
    market_close = datetime.combine(observed_ny.date(), time(16, 0), tzinfo=_NEW_YORK)
    columns = ["open", "high", "low", "close"]
    if "volume" in bars.columns:
        columns.append("volume")
    frame = bars.loc[:, columns].copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).tz_convert(
        _NEW_YORK
    )
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    finite_ohlc = pd.Series(True, index=frame.index)
    for column in ("open", "high", "low", "close"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = numeric
        finite_ohlc &= numeric.notna() & numeric.map(math.isfinite)
    finite_ohlc &= (
        (frame["open"] > 0.0)
        & (frame["high"] >= frame["low"])
        & (frame["high"] >= frame[["open", "close"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close"]].min(axis=1))
    )
    frame = frame.loc[finite_ohlc]
    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    completed_before = observed_ny - timedelta(minutes=timeframe_minutes)
    frame = frame[
        (frame.index >= market_open)
        & (frame.index < market_close)
        & (frame.index <= completed_before)
    ]
    if frame.empty:
        return None
    return _GenerationContext(
        symbol=normalized_symbol,
        observed_at=observed_at,
        session=observed_ny.date().isoformat(),
        option_expiry=observed_ny.strftime("%Y%m%d"),
        market_open=market_open,
        market_close=market_close,
        timeframe_minutes=timeframe_minutes,
        frame=frame,
    )


def _exact_window(
    context: _GenerationContext,
    *,
    start_at: datetime,
    minutes: int,
) -> pd.DataFrame | None:
    end_at = start_at + timedelta(minutes=minutes)
    expected = pd.DatetimeIndex(
        [
            start_at + timedelta(minutes=offset)
            for offset in range(0, minutes, context.timeframe_minutes)
        ]
    )
    if not expected.isin(context.frame.index).all():
        return None
    window = context.frame.loc[expected]
    if len(window) != len(expected) or end_at > context.observed_at.astimezone(_NEW_YORK):
        return None
    return window


def _atr_14(frame: pd.DataFrame) -> float | None:
    if len(frame) < _ATR_WINDOW:
        return None
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(
        alpha=1.0 / _ATR_WINDOW,
        adjust=False,
        min_periods=_ATR_WINDOW,
    ).mean()
    return _finite(atr.iloc[-1]) if not atr.empty else None


def _volume_values(frame: pd.DataFrame) -> pd.Series | None:
    if "volume" not in frame.columns:
        return None
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    return volume.where((volume >= 0) & volume.map(math.isfinite)).dropna()


def _vwap(frame: pd.DataFrame) -> float | None:
    volume = _volume_values(frame)
    if volume is None or volume.empty:
        return None
    positive = volume[volume > 0]
    if positive.empty:
        return None
    typical = (
        pd.to_numeric(frame.loc[positive.index, "high"], errors="coerce")
        + pd.to_numeric(frame.loc[positive.index, "low"], errors="coerce")
        + pd.to_numeric(frame.loc[positive.index, "close"], errors="coerce")
    ) / 3.0
    numerator = _finite((typical * positive).sum())
    denominator = _finite(positive.sum())
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def _mean_volume(frame: pd.DataFrame) -> float | None:
    values = _volume_values(frame)
    if values is None or values.empty:
        return None
    return _finite(values.mean())


def _window_descriptor(
    window: pd.DataFrame,
    *,
    timeframe_minutes: int,
) -> CausalWindow:
    start = cast(pd.Timestamp, window.index[0]).to_pydatetime()
    last_start = cast(pd.Timestamp, window.index[-1]).to_pydatetime()
    completed = last_start + timedelta(minutes=timeframe_minutes)
    return CausalWindow(
        start_at=start,
        end_at=completed,
        last_bar_started_at=last_start,
        last_bar_completed_at=completed,
        bar_count=len(window),
    )


def _momentum_measurements(
    window: pd.DataFrame,
    *,
    history_through_signal: pd.DataFrame,
    baseline_before_window: pd.DataFrame,
    direction: HypothesisDirection,
) -> MomentumMeasurements:
    open_price = float(window.iloc[0]["open"])
    close_price = float(window.iloc[-1]["close"])
    high_price = float(window["high"].max())
    low_price = float(window["low"].min())
    price_range = max(0.0, high_price - low_price)
    raw_return = (close_price - open_price) / open_price if open_price > 0.0 else 0.0
    directional_return = raw_return if direction == "bull" else -raw_return
    closes = pd.to_numeric(window["close"], errors="coerce")
    path = abs(float(closes.iloc[0]) - open_price) + float(closes.diff().abs().sum())
    efficiency = _unit(abs(close_price - open_price) / path) if path > 0.0 else 0.0
    close_location = (
        _unit(
            (close_price - low_price) / price_range
            if direction == "bull"
            else (high_price - close_price) / price_range
        )
        if price_range > 0.0
        else 0.0
    )
    atr = _atr_14(history_through_signal)
    directional_move = abs(close_price - open_price)
    return_atr = directional_move / atr if atr is not None and atr > 0.0 else None
    session_vwap = _vwap(history_through_signal)
    directional_vwap_distance: float | None = None
    vwap_aligned: bool | None = None
    if session_vwap is not None and session_vwap > 0.0:
        signed_distance = (
            close_price - session_vwap
            if direction == "bull"
            else session_vwap - close_price
        )
        directional_vwap_distance = signed_distance / session_vwap
        vwap_aligned = signed_distance >= 0.0
    volume = _volume_values(window)
    total_volume = _finite(volume.sum()) if volume is not None and not volume.empty else None
    mean_volume = _finite(volume.mean()) if volume is not None and not volume.empty else None
    baseline_mean = _mean_volume(baseline_before_window)
    relative_volume = (
        mean_volume / baseline_mean
        if mean_volume is not None and baseline_mean is not None and baseline_mean > 0.0
        else None
    )
    return MomentumMeasurements(
        open_price=open_price,
        close_price=close_price,
        high_price=high_price,
        low_price=low_price,
        return_pct=raw_return,
        directional_return_pct=max(0.0, directional_return),
        range_pct=price_range / open_price if open_price > 0.0 else 0.0,
        atr_14=atr,
        directional_return_atr_ratio=return_atr,
        directional_return_atr_normalized=_positive_ratio_to_unit(return_atr),
        directional_efficiency=efficiency,
        directional_close_location=close_location,
        vwap=session_vwap,
        directional_vwap_distance_pct=directional_vwap_distance,
        vwap_direction_aligned=vwap_aligned,
        total_volume=total_volume,
        mean_volume=mean_volume,
        relative_volume=relative_volume,
        relative_volume_normalized=_positive_ratio_to_unit(relative_volume),
    )


def _stable_id(
    *,
    generator: HypothesisKind,
    context: _GenerationContext,
    direction: HypothesisDirection,
    signal_at: datetime,
    anchor_at: datetime,
    parameter_version: str,
) -> str:
    identity = {
        "schema_version": _SCHEMA_VERSION,
        "generator": generator,
        "symbol": context.symbol,
        "session": context.session,
        "option_expiry": context.option_expiry,
        "direction": direction,
        "signal_at": _utc_iso(signal_at),
        "anchor_at": _utc_iso(anchor_at),
        "parameter_version": parameter_version,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hypothesis(
    *,
    context: _GenerationContext,
    generator: HypothesisKind,
    direction: HypothesisDirection,
    signal_at: datetime,
    anchor_at: datetime,
    expires_at: datetime,
    reference_price: float,
    invalidation_level: float,
    features: HypothesisFeatures,
) -> ShadowIntradayHypothesis | None:
    signal_utc = signal_at.astimezone(UTC)
    expiry_utc = expires_at.astimezone(UTC)
    if context.observed_at < signal_utc or context.observed_at > expiry_utc:
        return None
    return ShadowIntradayHypothesis(
        hypothesis_id=_stable_id(
            generator=generator,
            context=context,
            direction=direction,
            signal_at=signal_at,
            anchor_at=anchor_at,
            parameter_version=features.parameter_version,
        ),
        generator=generator,
        symbol=context.symbol,
        direction=direction,
        session=context.session,
        option_expiry=context.option_expiry,
        signal_at=signal_utc,
        observed_at=context.observed_at,
        causal_cutoff_at=signal_utc,
        thesis_expires_at=expiry_utc,
        reference_price=reference_price,
        invalidation_level=invalidation_level,
        features=features,
    )


def generate_opening_momentum_continuation(
    bars: pd.DataFrame,
    *,
    symbol: str,
    observed_at: datetime,
    config: HypothesisResearchConfig | None = None,
) -> tuple[ShadowIntradayHypothesis, ...]:
    """Describe the first fixed-window directional move without quality gating."""
    config = config or HypothesisResearchConfig()
    context = _prepare_context(
        bars,
        symbol=symbol,
        observed_at=observed_at,
        timeframe_minutes=config.timeframe_minutes,
    )
    if context is None:
        return ()
    window = _exact_window(
        context,
        start_at=context.market_open,
        minutes=config.opening_momentum_window_minutes,
    )
    if window is None:
        return ()
    open_price = float(window.iloc[0]["open"])
    close_price = float(window.iloc[-1]["close"])
    if close_price == open_price:
        return ()
    direction: HypothesisDirection = "bull" if close_price > open_price else "bear"
    descriptor = _window_descriptor(
        window,
        timeframe_minutes=context.timeframe_minutes,
    )
    history = context.frame.loc[context.frame.index < descriptor.end_at]
    midpoint = len(window) // 2
    first_half = _mean_volume(window.iloc[:midpoint])
    second_half = _mean_volume(window.iloc[midpoint:])
    second_half_ratio = (
        second_half / first_half
        if first_half is not None and second_half is not None and first_half > 0.0
        else None
    )
    features = OpeningMomentumFeatures(
        causal_window=descriptor,
        momentum=_momentum_measurements(
            window,
            history_through_signal=history,
            baseline_before_window=context.frame.iloc[0:0],
            direction=direction,
        ),
        opening_window_minutes=config.opening_momentum_window_minutes,
        thesis_lifetime_minutes=config.opening_momentum_lifetime_minutes,
        second_half_volume_ratio=second_half_ratio,
        second_half_volume_normalized=_positive_ratio_to_unit(second_half_ratio),
        parameter_version=config.parameter_version,
    )
    expires_at = min(
        descriptor.end_at + timedelta(minutes=config.opening_momentum_lifetime_minutes),
        context.market_close,
    )
    hypothesis = _hypothesis(
        context=context,
        generator="opening_momentum_continuation",
        direction=direction,
        signal_at=descriptor.end_at,
        anchor_at=descriptor.start_at,
        expires_at=expires_at,
        reference_price=close_price,
        invalidation_level=open_price,
        features=features,
    )
    return (hypothesis,) if hypothesis is not None else ()


def _bar_relative_volume(
    history: pd.DataFrame,
    *,
    timestamp: pd.Timestamp,
) -> tuple[float | None, float | None]:
    row = history.loc[timestamp]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    current = _finite(row["volume"]) if "volume" in row else None
    if current is not None and current < 0.0:
        current = None
    baseline = _mean_volume(history.loc[history.index < timestamp])
    relative = (
        current / baseline
        if current is not None and baseline is not None and baseline > 0.0
        else None
    )
    return current, relative


def generate_failed_breakout_reversals(
    bars: pd.DataFrame,
    *,
    symbol: str,
    observed_at: datetime,
    config: HypothesisResearchConfig | None = None,
) -> tuple[ShadowIntradayHypothesis, ...]:
    """Find a completed OR close breakout followed by a bounded OR re-entry."""
    config = config or HypothesisResearchConfig()
    context = _prepare_context(
        bars,
        symbol=symbol,
        observed_at=observed_at,
        timeframe_minutes=config.timeframe_minutes,
    )
    if context is None:
        return ()
    opening = _exact_window(
        context,
        start_at=context.market_open,
        minutes=config.opening_range_minutes,
    )
    if opening is None:
        return ()
    range_high = float(opening["high"].max())
    range_low = float(opening["low"].min())
    range_width = range_high - range_low
    if not range_width > 0.0:
        return ()
    range_end = context.market_open + timedelta(minutes=config.opening_range_minutes)
    post = context.frame.loc[context.frame.index >= range_end]
    records = list(post.iterrows())
    prior_close = float(opening.iloc[-1]["close"])
    results: list[ShadowIntradayHypothesis] = []
    for breakout_position, (breakout_key, breakout_bar) in enumerate(records):
        breakout_ts = cast(pd.Timestamp, breakout_key)
        breakout_close = float(breakout_bar["close"])
        failed_side: Literal["high", "low"] | None = None
        if breakout_close > range_high and prior_close <= range_high:
            failed_side = "high"
        elif breakout_close < range_low and prior_close >= range_low:
            failed_side = "low"
        prior_close = breakout_close
        if failed_side is None:
            continue
        reentry_end = min(
            len(records),
            breakout_position + config.failed_breakout_max_reentry_bars + 1,
        )
        for reentry_position in range(breakout_position + 1, reentry_end):
            reentry_key, reentry_bar = records[reentry_position]
            reentry_ts = cast(pd.Timestamp, reentry_key)
            reentry_close = float(reentry_bar["close"])
            if range_low <= reentry_close <= range_high:
                direction: HypothesisDirection = (
                    "bear" if failed_side == "high" else "bull"
                )
                signal_at = reentry_ts.to_pydatetime() + timedelta(
                    minutes=context.timeframe_minutes
                )
                history = context.frame.loc[context.frame.index <= reentry_ts]
                atr = _atr_14(history)
                range_atr = range_width / atr if atr is not None and atr > 0.0 else None
                breakout_open = float(breakout_bar["open"])
                breakout_high = float(breakout_bar["high"])
                breakout_low = float(breakout_bar["low"])
                breakout_range = max(0.0, breakout_high - breakout_low)
                if failed_side == "high":
                    breakout_extreme = breakout_high
                    close_displacement = breakout_close - range_high
                    extreme_excursion = breakout_high - range_high
                    rejection_wick = breakout_high - max(
                        breakout_open, breakout_close
                    )
                    reentry_depth = (range_high - reentry_close) / range_width
                    invalidation = breakout_high
                else:
                    breakout_extreme = breakout_low
                    close_displacement = range_low - breakout_close
                    extreme_excursion = range_low - breakout_low
                    rejection_wick = min(breakout_open, breakout_close) - breakout_low
                    reentry_depth = (reentry_close - range_low) / range_width
                    invalidation = breakout_low
                reentry_open = float(reentry_bar["open"])
                reentry_high = float(reentry_bar["high"])
                reentry_low = float(reentry_bar["low"])
                reentry_range = max(0.0, reentry_high - reentry_low)
                directional_body = max(
                    0.0,
                    reentry_close - reentry_open
                    if direction == "bull"
                    else reentry_open - reentry_close,
                )
                breakout_volume, breakout_rvol = _bar_relative_volume(
                    history,
                    timestamp=breakout_ts,
                )
                session_vwap = _vwap(history)
                vwap_aligned: bool | None = None
                if session_vwap is not None:
                    vwap_aligned = (
                        reentry_close >= session_vwap
                        if direction == "bull"
                        else reentry_close <= session_vwap
                    )
                causal_window = CausalWindow(
                    start_at=context.market_open,
                    end_at=signal_at,
                    last_bar_started_at=reentry_ts.to_pydatetime(),
                    last_bar_completed_at=signal_at,
                    bar_count=len(history),
                )
                features = FailedBreakoutFeatures(
                    causal_window=causal_window,
                    opening_range_high=range_high,
                    opening_range_low=range_low,
                    opening_range_width=range_width,
                    opening_range_atr_ratio=range_atr,
                    opening_range_minutes=config.opening_range_minutes,
                    failed_side=failed_side,
                    breakout_at=breakout_ts.to_pydatetime(),
                    breakout_close=breakout_close,
                    breakout_extreme=breakout_extreme,
                    breakout_close_displacement_or_ratio=max(
                        0.0, close_displacement / range_width
                    ),
                    breakout_extreme_excursion_or_ratio=max(
                        0.0, extreme_excursion / range_width
                    ),
                    breakout_rejection_wick_fraction=(
                        _unit(max(0.0, rejection_wick) / breakout_range)
                        if breakout_range > 0.0
                        else 0.0
                    ),
                    breakout_volume=breakout_volume,
                    breakout_relative_volume=breakout_rvol,
                    breakout_relative_volume_normalized=_positive_ratio_to_unit(
                        breakout_rvol
                    ),
                    reentry_at=reentry_ts.to_pydatetime(),
                    reentry_close=reentry_close,
                    bars_to_reentry=reentry_position - breakout_position,
                    max_reentry_bars=config.failed_breakout_max_reentry_bars,
                    thesis_lifetime_minutes=config.failed_breakout_lifetime_minutes,
                    reentry_depth_fraction=_unit(reentry_depth),
                    reentry_directional_body_fraction=(
                        _unit(directional_body / reentry_range)
                        if reentry_range > 0.0
                        else 0.0
                    ),
                    reentry_vwap=session_vwap,
                    reentry_vwap_direction_aligned=vwap_aligned,
                    parameter_version=config.parameter_version,
                )
                hypothesis = _hypothesis(
                    context=context,
                    generator="failed_breakout_reversal",
                    direction=direction,
                    signal_at=signal_at,
                    anchor_at=breakout_ts.to_pydatetime(),
                    expires_at=min(
                        signal_at
                        + timedelta(minutes=config.failed_breakout_lifetime_minutes),
                        context.market_close,
                    ),
                    reference_price=reentry_close,
                    invalidation_level=invalidation,
                    features=features,
                )
                if hypothesis is not None:
                    results.append(hypothesis)
                break
            if (
                failed_side == "high" and reentry_close < range_low
            ) or (
                failed_side == "low" and reentry_close > range_high
            ):
                break
    unique = {item.hypothesis_id: item for item in results}
    return tuple(sorted(unique.values(), key=lambda item: item.signal_at))


def generate_late_session_momentum(
    bars: pd.DataFrame,
    *,
    symbol: str,
    observed_at: datetime,
    config: HypothesisResearchConfig | None = None,
) -> tuple[ShadowIntradayHypothesis, ...]:
    """Describe a fixed late-session momentum window through the close."""
    config = config or HypothesisResearchConfig()
    context = _prepare_context(
        bars,
        symbol=symbol,
        observed_at=observed_at,
        timeframe_minutes=config.timeframe_minutes,
    )
    if context is None:
        return ()
    window_start = datetime.combine(
        context.market_open.date(),
        config.late_session_start,
        tzinfo=_NEW_YORK,
    )
    window = _exact_window(
        context,
        start_at=window_start,
        minutes=config.late_session_window_minutes,
    )
    if window is None:
        return ()
    open_price = float(window.iloc[0]["open"])
    close_price = float(window.iloc[-1]["close"])
    if close_price == open_price:
        return ()
    direction: HypothesisDirection = "bull" if close_price > open_price else "bear"
    descriptor = _window_descriptor(
        window,
        timeframe_minutes=context.timeframe_minutes,
    )
    history = context.frame.loc[context.frame.index < descriptor.end_at]
    baseline = context.frame.loc[context.frame.index < descriptor.start_at]
    features = LateSessionMomentumFeatures(
        causal_window=descriptor,
        momentum=_momentum_measurements(
            window,
            history_through_signal=history,
            baseline_before_window=baseline,
            direction=direction,
        ),
        minutes_from_open=(descriptor.start_at - context.market_open).total_seconds()
        / 60.0,
        minutes_to_close=(context.market_close - descriptor.end_at).total_seconds()
        / 60.0,
        window_minutes=config.late_session_window_minutes,
        configured_start_local=config.late_session_start.isoformat(),
        parameter_version=config.parameter_version,
    )
    hypothesis = _hypothesis(
        context=context,
        generator="late_session_momentum",
        direction=direction,
        signal_at=descriptor.end_at,
        anchor_at=descriptor.start_at,
        expires_at=context.market_close,
        reference_price=close_price,
        invalidation_level=open_price,
        features=features,
    )
    return (hypothesis,) if hypothesis is not None else ()


class HypothesisGenerator(Protocol):
    def __call__(
        self,
        bars: pd.DataFrame,
        *,
        symbol: str,
        observed_at: datetime,
        config: HypothesisResearchConfig | None = None,
    ) -> tuple[ShadowIntradayHypothesis, ...]: ...


# Public generator names are stable provenance keys.  Callers normally use
# ``generate_shadow_hypotheses``; the registry is exposed for selective shadow
# studies and is immutable to prevent runtime authority injection.
HYPOTHESIS_GENERATOR_REGISTRY: Mapping[
    HypothesisKind,
    HypothesisGenerator,
] = MappingProxyType(
    {
        "opening_momentum_continuation": generate_opening_momentum_continuation,
        "failed_breakout_reversal": generate_failed_breakout_reversals,
        "late_session_momentum": generate_late_session_momentum,
    }
)


def generate_shadow_hypotheses(
    bars: pd.DataFrame,
    *,
    symbol: str,
    observed_at: datetime,
    config: HypothesisResearchConfig | None = None,
    generators: tuple[HypothesisKind, ...] | None = None,
) -> tuple[ShadowIntradayHypothesis, ...]:
    """Run registered research generators and return stable causal outputs."""
    config = config or HypothesisResearchConfig()
    selected = generators or tuple(HYPOTHESIS_GENERATOR_REGISTRY)
    unknown = set(selected) - set(HYPOTHESIS_GENERATOR_REGISTRY)
    if unknown:
        raise ValueError(f"unknown shadow hypothesis generator(s): {sorted(unknown)}")
    results: list[ShadowIntradayHypothesis] = []
    for name in selected:
        generator = HYPOTHESIS_GENERATOR_REGISTRY[name]
        results.extend(
            generator(
                bars,
                symbol=symbol,
                observed_at=observed_at,
                config=config,
            )
        )
    unique = {item.hypothesis_id: item for item in results}
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (item.signal_at, item.generator, item.hypothesis_id),
        )
    )
