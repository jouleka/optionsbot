"""Simple moving average, ADX, and trend-regime classification.

Pure pandas implementations -- no TA-Lib dependency. ADX uses Wilder's
smoothing (RMA), matching the conventional formulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optionsbot.analysis.types import TrendRegime

_ADX_TRENDING_THRESHOLD = 25.0  # widely-used convention


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average. Returns a Series same length as input, with
    NaNs for the first ``window - 1`` rows."""
    return series.rolling(window=window).mean()


def _wilder_rma(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing (RMA) -- exponential MA with alpha = 1/window.

    pandas' ``ewm(alpha=1/window, adjust=False)`` matches Wilder exactly
    once the window has filled.
    """
    return series.ewm(alpha=1.0 / window, adjust=False).mean()


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder's Average Directional Index.

    Args:
        df: OHLCV DataFrame with high/low/close columns.
        window: smoothing window. Default 14 per the original Wilder formulation.

    Returns:
        Series of ADX values. NaN until window+1 bars are available.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    # True range
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Directional movement
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = _wilder_rma(tr, window)
    # Replace zeros with NaN (not pd.NA) so the dtype stays float64 and ewm
    # can still aggregate downstream. NaN naturally propagates through the
    # division and through subsequent _wilder_rma calls.
    plus_di = 100.0 * _wilder_rma(plus_dm, window) / atr.replace(0, np.nan)
    minus_di = 100.0 * _wilder_rma(minus_dm, window) / atr.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder_rma(dx, window)


def trend_regime(df: pd.DataFrame) -> TrendRegime:
    """Classify the trend as bull/neutral/bear x strong/weak using SMA + ADX.

    Direction is derived from the 20/50 SMA crossover state and slope of
    the 20-day SMA. Strength comes from the ADX (default window 14):
    strong if ADX >= 25, otherwise weak.

    For series too short to compute SMA50 or ADX, returns neutral/weak.
    """
    close = df["close"]
    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    a = adx(df, 14)

    sma20_last = sma20.iloc[-1] if not sma20.empty else None
    sma50_last = sma50.iloc[-1] if not sma50.empty else None
    adx_last = a.iloc[-1] if not a.empty else None

    direction: str = "neutral"
    if (
        sma20_last is not None
        and sma50_last is not None
        and not pd.isna(sma20_last)
        and not pd.isna(sma50_last)
    ):
        if sma20_last > sma50_last and sma20.iloc[-1] > sma20.iloc[-5]:
            direction = "bull"
        elif sma20_last < sma50_last and sma20.iloc[-1] < sma20.iloc[-5]:
            direction = "bear"

    strength: str = "weak"
    if adx_last is not None and not pd.isna(adx_last) and adx_last >= _ADX_TRENDING_THRESHOLD:
        strength = "strong"

    return TrendRegime(
        direction=direction,  # type: ignore[arg-type]
        strength=strength,  # type: ignore[arg-type]
        adx=float(adx_last) if adx_last is not None and not pd.isna(adx_last) else None,
        sma20=float(sma20_last) if sma20_last is not None and not pd.isna(sma20_last) else None,
        sma50=float(sma50_last) if sma50_last is not None and not pd.isna(sma50_last) else None,
    )
