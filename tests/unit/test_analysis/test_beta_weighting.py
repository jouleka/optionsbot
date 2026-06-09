"""Tests for beta-weighted portfolio delta (IBK-118)."""

from __future__ import annotations

import pandas as pd

from optionsbot.analysis.beta_weighting import beta


def _closes(returns: list[float], start: float = 100.0) -> pd.DataFrame:
    """A close series whose successive pct-changes equal `returns`."""
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1.0 + r))
    return pd.DataFrame({"close": closes})


_BENCH_RETS = [0.01, -0.012, 0.008, -0.005, 0.015, -0.009] * 6  # 36 varied daily returns


def test_beta_identical_series_is_one() -> None:
    bench = _closes(_BENCH_RETS)
    assert round(beta(bench, bench, window=36, min_obs=5), 6) == 1.0


def test_beta_double_moves_is_two() -> None:
    bench = _closes(_BENCH_RETS)
    sym = _closes([2.0 * r for r in _BENCH_RETS])
    assert round(beta(sym, bench, window=36, min_obs=5), 6) == 2.0


def test_beta_anti_correlated_is_negative() -> None:
    bench = _closes(_BENCH_RETS)
    sym = _closes([-1.0 * r for r in _BENCH_RETS])
    b = beta(sym, bench, window=36, min_obs=5)
    assert b is not None and round(b, 6) == -1.0


def test_beta_none_when_below_min_obs() -> None:
    bench = _closes(_BENCH_RETS)
    sym = _closes(_BENCH_RETS)
    # default min_obs is 30; only 36 returns here, but truncating to window=5 leaves 5 < 30.
    assert beta(sym, bench, window=5) is None


def test_beta_none_on_zero_benchmark_variance() -> None:
    bench = _closes([0.0] * 36)  # flat -> zero variance
    sym = _closes(_BENCH_RETS)
    assert beta(sym, bench, window=36, min_obs=5) is None


def test_beta_aligns_on_common_dates() -> None:
    bench = _closes(_BENCH_RETS)  # 37 closes
    bench.index = pd.date_range("2026-01-01", periods=37, freq="D")
    sym = _closes([1.5 * r for r in _BENCH_RETS])  # 37 closes
    sym.index = pd.date_range("2026-01-02", periods=37, freq="D")  # shifted by one day
    b = beta(sym, bench, window=36, min_obs=5)
    assert b is not None  # the date overlap still yields a finite beta
