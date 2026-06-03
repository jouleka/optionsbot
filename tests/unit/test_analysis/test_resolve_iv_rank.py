"""_resolve_iv_rank: HV-proxy when IV is warming up (IBK-94)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from optionsbot.analysis.types import IVRankResult
from optionsbot.analysis.view import _resolve_iv_rank


def _bars(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    closes = 100.0 * np.cumprod(1 + rng.normal(0, 0.015, n))
    return pd.DataFrame({"close": closes})


def test_uses_hv_proxy_when_warming_up() -> None:
    ivr = IVRankResult(rank=1.0, warming_up=True, sample_size=3)  # thin IV history
    rank, is_proxy = _resolve_iv_rank(ivr, _bars(120))
    assert is_proxy is True
    assert rank is not None and 0.0 <= rank <= 1.0


def test_uses_real_iv_rank_when_mature() -> None:
    ivr = IVRankResult(rank=0.42, warming_up=False, sample_size=200)
    rank, is_proxy = _resolve_iv_rank(ivr, _bars(120))
    assert is_proxy is False
    assert rank == 0.42


def test_falls_back_when_bars_too_short_for_hv() -> None:
    ivr = IVRankResult(rank=None, warming_up=True, sample_size=2)
    rank, is_proxy = _resolve_iv_rank(ivr, _bars(5))  # < window+1 -> no HV series
    assert is_proxy is False
    assert rank is None
