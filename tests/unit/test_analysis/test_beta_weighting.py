"""Tests for beta-weighted portfolio delta (IBK-118)."""

from __future__ import annotations

import pandas as pd

from optionsbot.analysis.beta_weighting import beta, beta_weighted_delta


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


def test_beta_weighted_delta_basic_sum() -> None:
    rows = [
        {"symbol": "SPY", "share_delta": 30.0, "spot": 600.0, "beta": 1.0},  # 18000
        {"symbol": "AAPL", "share_delta": 100.0, "spot": 200.0, "beta": 1.5},  # 30000
    ]
    out = beta_weighted_delta(rows, benchmark_spot=600.0)
    assert out["beta_weighted_dollar_delta"] == 48000.0
    assert out["dollar_per_1pct_spy"] == 480.0
    assert out["spy_equiv_shares"] == 80.0  # 48000 / 600
    assert out["underlyings_total"] == 2 and out["underlyings_covered"] == 2
    assert out["complete"] is True


def test_beta_weighted_delta_missing_beta_dings_coverage() -> None:
    rows = [
        {"symbol": "SPY", "share_delta": 30.0, "spot": 600.0, "beta": 1.0},
        {"symbol": "XYZ", "share_delta": 50.0, "spot": 20.0, "beta": None},  # no beta
    ]
    out = beta_weighted_delta(rows, benchmark_spot=600.0)
    assert out["underlyings_total"] == 2 and out["underlyings_covered"] == 1
    assert out["complete"] is False
    assert out["beta_weighted_dollar_delta"] == 18000.0  # only covered row


def test_beta_weighted_delta_no_benchmark_spot_drops_shares() -> None:
    rows = [{"symbol": "SPY", "share_delta": 30.0, "spot": 600.0, "beta": 1.0}]
    out = beta_weighted_delta(rows, benchmark_spot=None)
    assert out["spy_equiv_shares"] is None
    assert out["dollar_per_1pct_spy"] == 180.0


def test_beta_weighted_delta_neutral_row_not_weightable() -> None:
    rows = [
        {"symbol": "SPY", "share_delta": 0.0, "spot": 600.0, "beta": 1.0},  # neutral
        {"symbol": "AAPL", "share_delta": 100.0, "spot": 200.0, "beta": 1.5},
    ]
    out = beta_weighted_delta(rows, benchmark_spot=600.0)
    assert out["underlyings_total"] == 1 and out["underlyings_covered"] == 1
    assert out["complete"] is True


def test_beta_weighted_delta_all_missing_zero_coverage() -> None:
    rows = [{"symbol": "XYZ", "share_delta": 50.0, "spot": None, "beta": None}]
    out = beta_weighted_delta(rows, benchmark_spot=600.0)
    assert out["underlyings_total"] == 1 and out["underlyings_covered"] == 0
    assert out["complete"] is False
    assert out["beta_weighted_dollar_delta"] == 0.0
