"""Synthetic OHLCV fixtures for analysis tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    n_days: int = 252,
    start_price: float = 100.0,
    daily_vol: float = 0.01,
    drift: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Geometric-Brownian-motion synthetic OHLCV with deterministic seed.

    Returns a DataFrame indexed by date (most recent last) with columns
    [open, high, low, close, volume].
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=drift, scale=daily_vol, size=n_days)
    close = start_price * np.exp(np.cumsum(returns))
    # Open ~ previous close; High/Low ~ close +- noise
    open_ = np.concatenate([[start_price], close[:-1]])
    noise = rng.normal(loc=0.0, scale=daily_vol * 0.3, size=n_days) * close
    high = np.maximum(open_, close) + np.abs(noise)
    low = np.minimum(open_, close) - np.abs(noise)
    volume = rng.integers(low=1_000_000, high=10_000_000, size=n_days)
    idx = pd.bdate_range(end=date.today(), periods=n_days)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture()
def constant_ohlcv() -> pd.DataFrame:
    """Flat-price DataFrame -- HV should be 0, ADX should drop, etc."""
    idx = pd.bdate_range(end=date.today(), periods=60)
    price = 100.0
    return pd.DataFrame(
        {
            "open": [price] * 60,
            "high": [price] * 60,
            "low": [price] * 60,
            "close": [price] * 60,
            "volume": [1_000_000] * 60,
        },
        index=idx,
    )
