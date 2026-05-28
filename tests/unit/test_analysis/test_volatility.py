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


# ---------------------------------------------------------------------------
# Property-style tests (IBK-68): invariants that should hold across inputs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 10.0, 100.0])
def test_hv_is_scale_invariant(scale: float) -> None:
    """HV is computed on log returns, so multiplying all closes by a constant
    leaves the log returns (and therefore HV) unchanged."""
    base = pd.Series([100.0 + i * 0.5 for i in range(30)])
    scaled = base * scale
    hv_base = historical_volatility(base, window=20)
    hv_scaled = historical_volatility(scaled, window=20)
    assert hv_scaled == pytest.approx(hv_base, rel=1e-9)


@pytest.mark.parametrize("size", [21, 50, 100, 252])
def test_hv_is_non_negative(size: int) -> None:
    """HV is a standard deviation; cannot be negative for any non-empty input."""
    rng = np.random.default_rng(42)
    closes = pd.Series(100 + rng.normal(0, 1, size=size).cumsum())
    hv = historical_volatility(closes, window=20)
    assert hv >= 0.0


def test_hv_monotone_in_noise() -> None:
    """Higher-noise series should produce strictly higher HV than lower-noise."""
    rng = np.random.default_rng(7)
    low_noise = pd.Series(100 + rng.normal(0, 0.5, size=100).cumsum())
    high_noise = pd.Series(100 + rng.normal(0, 5.0, size=100).cumsum())
    hv_low = historical_volatility(low_noise, window=50)
    hv_high = historical_volatility(high_noise, window=50)
    assert hv_high > hv_low


@pytest.mark.parametrize("iv_multiplier", [0.5, 1.0, 2.0, 5.0])
def test_iv_hv_ratio_scales_linearly_with_iv(iv_multiplier: float) -> None:
    """For fixed HV, the ratio is linear in IV."""
    base_iv = 0.20
    base_hv = 0.10
    base_ratio = iv_hv_ratio(iv=base_iv, hv=base_hv)
    scaled_ratio = iv_hv_ratio(iv=base_iv * iv_multiplier, hv=base_hv)
    assert scaled_ratio == pytest.approx(base_ratio * iv_multiplier)


@pytest.mark.parametrize("hv_multiplier", [0.5, 1.0, 2.0, 5.0])
def test_iv_hv_ratio_inverse_in_hv(hv_multiplier: float) -> None:
    """For fixed IV, the ratio is inversely proportional to HV."""
    iv = 0.30
    base_hv = 0.20
    base_ratio = iv_hv_ratio(iv=iv, hv=base_hv)
    scaled_ratio = iv_hv_ratio(iv=iv, hv=base_hv * hv_multiplier)
    assert scaled_ratio == pytest.approx(base_ratio / hv_multiplier)


@pytest.mark.parametrize("spot", [10.0, 100.0, 400.0, 5000.0])
def test_expected_move_scales_linearly_with_spot(spot: float) -> None:
    """Doubling spot doubles expected move (atm_iv and dte held constant)."""
    em = expected_move(spot=spot, atm_iv=0.20, dte=30)
    em_2x = expected_move(spot=spot * 2, atm_iv=0.20, dte=30)
    assert em_2x == pytest.approx(em * 2)


@pytest.mark.parametrize("dte_pair", [(1, 4), (5, 20), (10, 40), (16, 64)])
def test_expected_move_scales_with_sqrt_dte(dte_pair: tuple[int, int]) -> None:
    """Quadrupling DTE doubles expected move (since EM ~ sqrt(dte))."""
    short_dte, long_dte = dte_pair
    em_short = expected_move(spot=100.0, atm_iv=0.25, dte=short_dte)
    em_long = expected_move(spot=100.0, atm_iv=0.25, dte=long_dte)
    # long_dte = 4 * short_dte, so em_long = 2 * em_short
    assert em_long == pytest.approx(em_short * 2, rel=1e-9)
