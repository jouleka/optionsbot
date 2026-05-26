"""Tests for SMA, ADX, and trend regime."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from optionsbot.analysis.technicals import adx, sma, trend_regime
from optionsbot.analysis.types import TrendRegime


def test_sma_returns_simple_moving_average() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(series, window=3)
    # Last value should be mean of [3, 4, 5] = 4
    assert result.iloc[-1] == pytest.approx(4.0)
    # Second value should be NaN (window not yet full)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])


def test_sma_window_larger_than_series_yields_all_nan() -> None:
    series = pd.Series([1.0, 2.0, 3.0])
    result = sma(series, window=10)
    assert result.isna().all()


def test_adx_of_constant_series_is_low(constant_ohlcv: pd.DataFrame) -> None:
    # No directional movement -> ADX should be very low (close to 0 or NaN).
    a = adx(constant_ohlcv, window=14)
    last = a.iloc[-1]
    assert pd.isna(last) or last < 5.0


def test_adx_of_strong_uptrend_is_above_25() -> None:
    # Strictly increasing price -> ADX should rise above the conventional
    # 25 "trending" threshold.
    idx = pd.bdate_range(end=date.today(), periods=100)
    close = pd.Series(np.linspace(100, 200, 100), index=idx)
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000_000,
        },
        index=idx,
    )
    a = adx(df, window=14)
    assert a.iloc[-1] > 25.0


def test_trend_regime_strong_bull_for_rising_series() -> None:
    idx = pd.bdate_range(end=date.today(), periods=100)
    close = pd.Series(np.linspace(100, 200, 100), index=idx)
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000_000,
        },
        index=idx,
    )
    regime = trend_regime(df)
    assert isinstance(regime, TrendRegime)
    assert regime.direction == "bull"
    assert regime.strength == "strong"


def test_trend_regime_strong_bear_for_falling_series() -> None:
    idx = pd.bdate_range(end=date.today(), periods=100)
    close = pd.Series(np.linspace(200, 100, 100), index=idx)
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000_000,
        },
        index=idx,
    )
    regime = trend_regime(df)
    assert regime.direction == "bear"
    assert regime.strength == "strong"


def test_trend_regime_neutral_for_flat_series(constant_ohlcv: pd.DataFrame) -> None:
    regime = trend_regime(constant_ohlcv)
    assert regime.direction == "neutral"
    assert regime.strength == "weak"
