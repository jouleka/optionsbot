"""Tests for IV rank computation with bootstrap-period flag."""

from __future__ import annotations

import pandas as pd
import pytest

from optionsbot.analysis.iv_rank import iv_rank
from optionsbot.analysis.types import IVRankResult


def test_iv_rank_at_max_of_window() -> None:
    hist = pd.Series([0.10, 0.15, 0.20, 0.25])
    result = iv_rank(current_iv=0.25, history=hist)
    assert isinstance(result, IVRankResult)
    assert result.rank == pytest.approx(1.0)
    assert result.warming_up is True  # < 30 samples
    assert result.sample_size == 4


def test_iv_rank_at_min_of_window() -> None:
    hist = pd.Series([0.10, 0.15, 0.20, 0.25])
    result = iv_rank(current_iv=0.10, history=hist)
    assert result.rank == pytest.approx(0.0)


def test_iv_rank_mid_range() -> None:
    hist = pd.Series([0.10, 0.20])  # range [0.10, 0.20], width 0.10
    result = iv_rank(current_iv=0.15, history=hist)
    assert result.rank == pytest.approx(0.5)


def test_iv_rank_warming_up_flag_clears_after_30_samples() -> None:
    hist = pd.Series([0.10 + 0.001 * i for i in range(31)])
    result = iv_rank(current_iv=0.15, history=hist)
    assert result.warming_up is False
    assert result.sample_size == 31


def test_iv_rank_returns_none_when_history_empty() -> None:
    result = iv_rank(current_iv=0.20, history=pd.Series(dtype=float))
    assert result.rank is None
    assert result.warming_up is True
    assert result.sample_size == 0


def test_iv_rank_returns_none_when_history_is_constant() -> None:
    # All values equal -> max == min, can't normalise.
    hist = pd.Series([0.20] * 10)
    result = iv_rank(current_iv=0.20, history=hist)
    assert result.rank is None
    assert result.warming_up is True


def test_iv_rank_clamps_above_range() -> None:
    hist = pd.Series([0.10, 0.20])
    result = iv_rank(current_iv=0.30, history=hist)
    assert result.rank == pytest.approx(1.0)


def test_iv_rank_clamps_below_range() -> None:
    hist = pd.Series([0.10, 0.20])
    result = iv_rank(current_iv=0.05, history=hist)
    assert result.rank == pytest.approx(0.0)


def test_iv_rank_uses_last_252_days_by_default() -> None:
    # 300 days of history; only the last 252 should count.
    hist = pd.Series([0.50] * 48 + [0.10, 0.20] * 126)  # first 48 are outliers
    result = iv_rank(current_iv=0.15, history=hist)
    # If we used the full series, max=0.50 -> rank would be 0.125.
    # With last-252 only, range is [0.10, 0.20] -> rank 0.5.
    assert result.rank == pytest.approx(0.5)
    assert result.sample_size == 252
