"""Realized historical volatility, IV/HV ratio, and expected move.

Pure functions; no I/O. Inputs are pandas Series of close prices or
bare floats. Outputs are floats (NaN when undefined).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

_ANNUALIZATION_FACTOR = 252  # trading days per year


def historical_volatility(closes: pd.Series, window: int = 20) -> float:
    """Annualized stdev of daily log returns over the trailing window.

    Returns NaN when there aren't enough returns (i.e., window + 1 closes).
    """
    if len(closes) < window + 1:
        return float("nan")
    log_returns = pd.Series(np.log(closes / closes.shift(1))).dropna()
    tail = log_returns.tail(window)
    if len(tail) < window:
        return float("nan")
    return float(tail.std(ddof=1) * math.sqrt(_ANNUALIZATION_FACTOR))


def historical_volatility_series(closes: pd.Series, window: int = 20) -> pd.Series:
    """Rolling annualized realized vol (stdev of daily log returns).

    Returns a Series aligned to ``closes``; the first ``window`` entries are NaN
    (insufficient returns). The last non-NaN value equals
    ``historical_volatility(closes, window)``. Use it to rank current HV against
    its own history (e.g. an HV-rank proxy while IV history is thin).
    """
    log_returns: pd.Series = pd.Series(
        np.log(closes / closes.shift(1)), index=closes.index
    )
    return pd.Series(
        log_returns.rolling(window).std(ddof=1) * math.sqrt(_ANNUALIZATION_FACTOR),
        index=closes.index,
    )


def iv_hv_ratio(iv: float, hv: float) -> float:
    """Ratio of implied to historical volatility. NaN if hv == 0 or either is NaN."""
    if math.isnan(iv) or math.isnan(hv):
        return float("nan")
    if hv == 0.0:
        return float("nan")
    return iv / hv


def expected_move(spot: float, atm_iv: float, dte: int) -> float:
    """One-sigma expected move over DTE days. NaN for negative DTE."""
    if dte < 0:
        return float("nan")
    if dte == 0:
        return 0.0
    return spot * atm_iv * math.sqrt(dte / 365)
