"""Tests for market view synthesis."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from optionsbot.analysis.types import (
    EarningsInfo,
    IVRankResult,
    MarketView,
    TrendRegime,
)
from optionsbot.analysis.view import infer_view


def _make_iv_history(level: float = 0.20, n: int = 60) -> pd.Series:
    """Constant IV history -> iv_rank is None (no range) by design.

    For tests that want a specific rank, patch iv_rank instead.
    """
    return pd.Series([level] * n)


def test_view_combines_components() -> None:
    with patch("optionsbot.analysis.view.trend_regime") as mock_tr, \
         patch("optionsbot.analysis.view.iv_rank") as mock_ivr, \
         patch("optionsbot.analysis.view.next_earnings") as mock_ne:
        mock_tr.return_value = TrendRegime(
            direction="bull", strength="strong", adx=30.0, sma20=110.0, sma50=105.0
        )
        mock_ivr.return_value = IVRankResult(rank=0.75, warming_up=False, sample_size=200)
        mock_ne.return_value = EarningsInfo(next_date=None, source="unknown")
        view = infer_view(
            symbol="SPY",
            bars=pd.DataFrame(),
            current_atm_iv=0.25,
            atm_iv_history=_make_iv_history(),
            earnings_window_days=14,
        )
    assert isinstance(view, MarketView)
    assert view.direction == "bull"
    assert view.direction_strength == "strong"
    assert view.iv_regime == "high"
    assert view.iv_rank_value == pytest.approx(0.75)
    assert view.earnings_in_window is False
    assert view.warming_up is False


def test_iv_regime_thresholds_low_neutral_high() -> None:
    cases = [
        (0.0, "low"),
        (0.29, "low"),
        (0.30, "neutral"),
        (0.50, "neutral"),
        (0.59, "neutral"),
        (0.60, "high"),
        (1.0, "high"),
    ]
    for rank, expected_regime in cases:
        with patch("optionsbot.analysis.view.trend_regime") as mock_tr, \
             patch("optionsbot.analysis.view.iv_rank") as mock_ivr, \
             patch("optionsbot.analysis.view.next_earnings") as mock_ne:
            mock_tr.return_value = TrendRegime("neutral", "weak", None, None, None)
            mock_ivr.return_value = IVRankResult(rank=rank, warming_up=False, sample_size=200)
            mock_ne.return_value = EarningsInfo(next_date=None, source="unknown")
            view = infer_view("SPY", pd.DataFrame(), 0.25, _make_iv_history())
        assert view.iv_regime == expected_regime, f"rank={rank} -> expected {expected_regime}"


def test_iv_regime_neutral_when_rank_is_none() -> None:
    with patch("optionsbot.analysis.view.trend_regime") as mock_tr, \
         patch("optionsbot.analysis.view.iv_rank") as mock_ivr, \
         patch("optionsbot.analysis.view.next_earnings") as mock_ne:
        mock_tr.return_value = TrendRegime("neutral", "weak", None, None, None)
        mock_ivr.return_value = IVRankResult(rank=None, warming_up=True, sample_size=0)
        mock_ne.return_value = EarningsInfo(next_date=None, source="unknown")
        view = infer_view("SPY", pd.DataFrame(), 0.25, _make_iv_history())
    assert view.iv_regime == "neutral"
    assert view.iv_rank_value is None
    assert view.warming_up is True


def test_earnings_in_window_propagates() -> None:
    from datetime import date, timedelta
    with patch("optionsbot.analysis.view.trend_regime") as mock_tr, \
         patch("optionsbot.analysis.view.iv_rank") as mock_ivr, \
         patch("optionsbot.analysis.view.next_earnings") as mock_ne:
        mock_tr.return_value = TrendRegime("neutral", "weak", None, None, None)
        mock_ivr.return_value = IVRankResult(rank=0.5, warming_up=False, sample_size=200)
        mock_ne.return_value = EarningsInfo(
            next_date=date.today() + timedelta(days=10), source="yfinance"
        )
        view = infer_view(
            "SPY", pd.DataFrame(), 0.25, _make_iv_history(), earnings_window_days=14
        )
    assert view.earnings_in_window is True


def test_earnings_outside_window_propagates_false() -> None:
    from datetime import date, timedelta
    with patch("optionsbot.analysis.view.trend_regime") as mock_tr, \
         patch("optionsbot.analysis.view.iv_rank") as mock_ivr, \
         patch("optionsbot.analysis.view.next_earnings") as mock_ne:
        mock_tr.return_value = TrendRegime("neutral", "weak", None, None, None)
        mock_ivr.return_value = IVRankResult(rank=0.5, warming_up=False, sample_size=200)
        mock_ne.return_value = EarningsInfo(
            next_date=date.today() + timedelta(days=30), source="yfinance"
        )
        view = infer_view(
            "SPY", pd.DataFrame(), 0.25, _make_iv_history(), earnings_window_days=14
        )
    assert view.earnings_in_window is False
