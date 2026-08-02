"""Deterministic opening-range breakout and fair-value-gap retest signal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, time, timedelta
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

import pandas as pd

_NEW_YORK = ZoneInfo("America/New_York")
Direction = Literal["bull", "bear"]


@dataclass(frozen=True, slots=True)
class OpeningRangeFVGSignal:
    signal_id: str
    session: str
    timeframe_minutes: int
    direction: Direction
    opening_range_high: float
    opening_range_low: float
    breakout_ts: datetime
    fvg_formed_ts: datetime
    fvg_low: float
    fvg_high: float
    respected_ts: datetime
    entry_underlying_price: float
    stop_pct: float
    target_r: float
    target_pct: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("breakout_ts", "fvg_formed_ts", "respected_ts"):
            result[key] = result[key].astimezone(UTC).isoformat()
        result["status"] = "entry_confirmed"
        result["source"] = "trusted_daemon"
        return result


def _bars_in_new_york(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    if bars.empty or not required.issubset(bars.columns):
        return pd.DataFrame(columns=sorted(required))
    result = bars.loc[:, ["open", "high", "low", "close"]].copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index, utc=True))
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]
    return result.tz_convert(_NEW_YORK)


def detect_opening_range_fvg(
    bars: pd.DataFrame,
    *,
    symbol: str,
    now: datetime,
    timeframe_minutes: int = 1,
    opening_range_minutes: int = 10,
    entry_window_minutes: int = 90,
    stop_pct: float = 0.15,
    target_r_min: float = 1.5,
    target_r_max: float = 2.0,
) -> OpeningRangeFVGSignal | None:
    """Return the newest confirmed same-session ORB/FVG retest, if any.

    Bar timestamps are interpreted as bar starts. The opening range contains
    exactly the bars starting at 09:30 through 09:39 for the default one-minute
    setup. A breakout requires a completed candle close outside that range.
    The FVG must form at or after that breakout, and a later candle must enter
    the gap without violating its far edge and close back through the near edge
    in the breakout direction.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    now_ny = now.astimezone(_NEW_YORK)
    frame = _bars_in_new_york(bars)
    if frame.empty:
        return None
    session = now_ny.date()
    frame_index = cast(pd.DatetimeIndex, frame.index)
    frame = frame.loc[[stamp.date() == session for stamp in frame_index]]
    # Never reason from an in-progress candle.
    completed_before = now_ny - timedelta(minutes=timeframe_minutes)
    frame = frame[frame.index <= completed_before]
    if frame.empty:
        return None

    market_open = datetime.combine(session, time(9, 30), tzinfo=_NEW_YORK)
    range_end = market_open + timedelta(minutes=opening_range_minutes)
    entry_end = market_open + timedelta(minutes=entry_window_minutes)
    opening = frame[(frame.index >= market_open) & (frame.index < range_end)]
    expected_bars = opening_range_minutes // timeframe_minutes
    expected_index = {
        market_open + timedelta(minutes=offset)
        for offset in range(0, opening_range_minutes, timeframe_minutes)
    }
    if len(opening) < expected_bars or not expected_index.issubset(set(opening.index)):
        return None
    range_high = float(opening["high"].max())
    range_low = float(opening["low"].min())
    if not range_high > range_low:
        return None

    eligible = frame[(frame.index >= range_end) & (frame.index < entry_end)]
    if len(eligible) < 4:
        return None
    breakout_position: int | None = None
    direction: Direction | None = None
    for position, (_, bar) in enumerate(eligible.iterrows()):
        close = float(bar["close"])
        if close > range_high:
            breakout_position, direction = position, "bull"
            break
        if close < range_low:
            breakout_position, direction = position, "bear"
            break
    if breakout_position is None or direction is None:
        return None

    records = list(eligible.iterrows())
    signals: list[OpeningRangeFVGSignal] = []
    range_size = range_high - range_low
    breakout_key, breakout_bar = records[breakout_position]
    breakout_ts = cast(pd.Timestamp, breakout_key)
    # The pattern cannot be confirmed by the breakout candle itself. The
    # earliest valid third candle is the candle after a later breakout (where
    # the breakout is the displacement/middle candle), or two candles after a
    # breakout in the first post-range slot.
    for formed in range(max(2, breakout_position + 1), len(records)):
        _, first = records[formed - 2]
        formed_key, third = records[formed]
        formed_ts = cast(pd.Timestamp, formed_key)
        if direction == "bull":
            gap_low = float(first["high"])
            gap_high = float(third["low"])
            if gap_high <= gap_low:
                continue
        else:
            gap_low = float(third["high"])
            gap_high = float(first["low"])
            if gap_high <= gap_low:
                continue

        for retest in range(formed + 1, len(records)):
            retest_key, bar = records[retest]
            retest_ts = cast(pd.Timestamp, retest_key)
            low = float(bar["low"])
            high = float(bar["high"])
            open_ = float(bar["open"])
            close = float(bar["close"])
            overlaps = low <= gap_high and high >= gap_low
            if direction == "bull":
                if low < gap_low:
                    break
                respected = overlaps and close >= gap_high and close > open_
            else:
                if high > gap_high:
                    break
                respected = overlaps and close <= gap_low and close < open_
            if not respected:
                continue

            breakout_body = abs(float(breakout_bar["close"]) - float(breakout_bar["open"]))
            retest_range = high - low
            retest_body_fraction = abs(close - open_) / retest_range if retest_range > 0 else 0.0
            strong = breakout_body >= 0.25 * range_size and retest_body_fraction >= 0.50
            target_r = target_r_max if strong else target_r_min
            signal_id = (
                f"{session.isoformat()}:{symbol.upper()}:{direction}:"
                f"{formed_ts.isoformat()}:{retest_ts.isoformat()}"
            )
            signals.append(
                OpeningRangeFVGSignal(
                    signal_id=signal_id,
                    session=session.isoformat(),
                    timeframe_minutes=timeframe_minutes,
                    direction=direction,
                    opening_range_high=range_high,
                    opening_range_low=range_low,
                    breakout_ts=breakout_ts.to_pydatetime(),
                    fvg_formed_ts=formed_ts.to_pydatetime(),
                    fvg_low=gap_low,
                    fvg_high=gap_high,
                    respected_ts=retest_ts.to_pydatetime(),
                    entry_underlying_price=close,
                    stop_pct=stop_pct,
                    target_r=target_r,
                    target_pct=stop_pct * target_r,
                )
            )
            break
    return max(signals, key=lambda signal: signal.respected_ts) if signals else None
