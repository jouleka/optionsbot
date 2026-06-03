"""Rolling realized-vol series (IBK-94)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from optionsbot.analysis.volatility import historical_volatility, historical_volatility_series


def test_series_length_matches_input() -> None:
    closes = pd.Series(np.linspace(100.0, 120.0, 60))
    s = historical_volatility_series(closes, window=20)
    assert len(s) == len(closes)


def test_series_short_history_all_nan() -> None:
    closes = pd.Series([100.0, 101.0, 102.0])  # < window + 1
    s = historical_volatility_series(closes, window=20)
    assert s.dropna().empty


def test_series_constant_prices_zero_vol() -> None:
    closes = pd.Series([100.0] * 40)
    s = historical_volatility_series(closes, window=20)
    assert float(s.dropna().iloc[-1]) == 0.0


def test_series_last_value_matches_scalar_hv() -> None:
    rng = np.random.default_rng(0)
    closes = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, 80)))
    s = historical_volatility_series(closes, window=20)
    assert float(s.dropna().iloc[-1]) == pytest.approx(historical_volatility(closes, window=20))


def test_series_is_annualized_positive() -> None:
    rng = np.random.default_rng(1)
    closes = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.02, 100)))
    s = historical_volatility_series(closes, window=20).dropna()
    assert (s > 0).all()
    assert not math.isnan(float(s.iloc[-1]))
