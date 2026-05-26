"""Tests for historical volatility, IV/HV ratio, and expected move."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from optionsbot.analysis.volatility import (
    expected_move,
    historical_volatility,
    iv_hv_ratio,
)


def test_hv_of_constant_series_is_zero(constant_ohlcv: pd.DataFrame) -> None:
    hv = historical_volatility(constant_ohlcv["close"], window=20)
    assert hv == pytest.approx(0.0, abs=1e-10)


def test_hv_matches_manual_annualization() -> None:
    # Hand-built: 21 closes with daily log returns of +0.02 and -0.02 alternating.
    # closes[i] = 100*exp(0.01*(-1)^i), so log(c[i]/c[i-1]) = ±0.02.
    closes = pd.Series([100 * math.exp(0.01 * (-1) ** i) for i in range(21)])
    hv = historical_volatility(closes, window=20)
    # log returns are alternating +/- 0.02, std of those is 0.02 (approx).
    # Annualized by sqrt(252).
    expected = 0.02 * math.sqrt(252)
    assert hv == pytest.approx(expected, rel=0.05)


def test_hv_returns_nan_when_window_too_short() -> None:
    closes = pd.Series([100.0, 101.0, 102.0])
    hv = historical_volatility(closes, window=20)
    assert math.isnan(hv)


def test_iv_hv_ratio_typical_values() -> None:
    assert iv_hv_ratio(iv=0.25, hv=0.20) == pytest.approx(1.25)
    assert iv_hv_ratio(iv=0.15, hv=0.20) == pytest.approx(0.75)


def test_iv_hv_ratio_returns_nan_for_zero_hv() -> None:
    result = iv_hv_ratio(iv=0.25, hv=0.0)
    assert math.isnan(result)


def test_iv_hv_ratio_returns_nan_for_nan_inputs() -> None:
    assert math.isnan(iv_hv_ratio(iv=float("nan"), hv=0.2))
    assert math.isnan(iv_hv_ratio(iv=0.2, hv=float("nan")))


def test_expected_move_one_sigma() -> None:
    # spot=400, IV=0.20, DTE=45 -> 400 * 0.20 * sqrt(45/365) ~ 28.0
    em = expected_move(spot=400.0, atm_iv=0.20, dte=45)
    manual = 400.0 * 0.20 * math.sqrt(45 / 365)
    assert em == pytest.approx(manual)


def test_expected_move_zero_dte_is_zero() -> None:
    assert expected_move(spot=400.0, atm_iv=0.20, dte=0) == pytest.approx(0.0)


def test_expected_move_negative_dte_returns_nan() -> None:
    assert math.isnan(expected_move(spot=400.0, atm_iv=0.20, dte=-1))


def test_hv_default_window_is_20() -> None:
    # Calling with no window should equal calling with window=20.
    closes = pd.Series(np.random.default_rng(1).normal(loc=100, scale=1, size=30))
    assert historical_volatility(closes) == historical_volatility(closes, window=20)
