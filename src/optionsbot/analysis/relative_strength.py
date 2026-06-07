"""Relative strength vs a benchmark (IBK-109).

Pure helper: a symbol's window return minus a benchmark's, from daily bars.
Context for the daily_brief reasoning layer -- NOT a scoring input.
"""

from __future__ import annotations

import pandas as pd


def _window_return(bars: pd.DataFrame, window: int) -> float | None:
    """Total return over the last ``window`` trading days, or None if too short."""
    if "close" not in bars or len(bars) < window + 1:
        return None
    closes = bars["close"]
    start = closes.iloc[-(window + 1)]
    end = closes.iloc[-1]
    if pd.isna(start) or pd.isna(end) or start <= 0:
        return None
    return float(end / start - 1.0)


def relative_strength(
    symbol_bars: pd.DataFrame, benchmark_bars: pd.DataFrame, window: int = 20
) -> float | None:
    """``symbol_return - benchmark_return`` over the last ``window`` trading days.

    None if either series has fewer than ``window + 1`` closes (or a degenerate
    start price). Both frames are daily bars with a ``close`` column.
    """
    sym = _window_return(symbol_bars, window)
    bench = _window_return(benchmark_bars, window)
    if sym is None or bench is None:
        return None
    return sym - bench
