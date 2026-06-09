"""Beta-weighted portfolio delta (IBK-118).

Pure helpers mirroring ``analysis/relative_strength.py``:
- ``beta`` estimates an underlying's beta vs a benchmark from daily bars.
- ``beta_weighted_delta`` aggregates per-underlying share-deltas into one
  SPY-comparable risk number.

Reporting only -- NOT a scoring input (see the no-manufactured-confidence principle).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Below this many overlapping daily returns, a beta estimate is too noisy to trust;
# we return None (excluded from weighting) rather than fake a number.
_MIN_BETA_OBS = 30


def beta(
    symbol_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    window: int = 252,
    min_obs: int = _MIN_BETA_OBS,
) -> float | None:
    """``cov(r_sym, r_bench) / var(r_bench)`` on date-aligned daily simple returns.

    Returns over each series' own consecutive closes, then aligned on common dates
    (handles halts / unequal listing length). Uses the last ``min(window, available)``
    paired returns. None if fewer than ``min_obs`` paired returns, if a ``close``
    column is missing, or if benchmark variance is zero.
    """
    if "close" not in symbol_bars or "close" not in benchmark_bars:
        return None
    s_ret = symbol_bars["close"].astype(float).pct_change(fill_method=None)
    b_ret = benchmark_bars["close"].astype(float).pct_change(fill_method=None)
    rets = pd.DataFrame({"s": s_ret, "b": b_ret}).dropna()
    if len(rets) > window:
        rets = rets.iloc[-window:]
    if len(rets) < min_obs:
        return None
    var_b = float(rets["b"].var())
    if var_b == 0.0 or pd.isna(var_b):
        return None
    return float(rets["s"].cov(rets["b"]) / var_b)
