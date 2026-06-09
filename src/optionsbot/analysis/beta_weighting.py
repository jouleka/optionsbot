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


def beta_weighted_delta(
    per_underlying: list[dict[str, Any]], benchmark_spot: float | None
) -> dict[str, Any]:
    """Aggregate per-underlying share-deltas into one benchmark-comparable number.

    ``per_underlying`` rows: ``{"symbol", "share_delta", "spot", "beta"}`` (spot/beta may
    be None). A row is *weightable* iff its ``share_delta`` is non-zero (a delta-neutral
    group contributes 0 regardless of beta and must not ding coverage); *covered* iff it
    is weightable and has both beta and spot. ``S = sum(beta * share_delta * spot)`` over
    covered rows. Pure.
    """
    s = 0.0
    total = 0
    covered = 0
    for row in per_underlying:
        share_delta = row.get("share_delta")
        if not share_delta:  # None or 0.0 -> not weightable
            continue
        total += 1
        b = row.get("beta")
        spot = row.get("spot")
        if b is None or spot is None:
            continue
        covered += 1
        s += b * share_delta * spot
    return {
        "beta_weighted_dollar_delta": s,
        "dollar_per_1pct_spy": s * 0.01,
        "spy_equiv_shares": (s / benchmark_spot) if benchmark_spot else None,
        "underlyings_total": total,
        "underlyings_covered": covered,
        "complete": covered == total,
    }
