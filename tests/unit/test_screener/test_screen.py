"""Pure screener ranking + metrics (IBK-95)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from optionsbot.screener.screen import (
    ScreenMetrics,
    rank_candidates,
    screen_metrics,
)


def test_rank_gates_on_liquidity_and_sorts_by_hv_rank() -> None:
    metrics = {
        "AAA": ScreenMetrics(dollar_volume=10_000_000.0, hv_rank=0.9),
        "BBB": ScreenMetrics(dollar_volume=10_000_000.0, hv_rank=0.5),
        "LOW": ScreenMetrics(dollar_volume=1_000.0, hv_rank=0.99),  # gated out
        "NOR": ScreenMetrics(dollar_volume=10_000_000.0, hv_rank=None),  # no rank -> skip
    }
    out = rank_candidates(metrics, min_dollar_volume=5_000_000.0)
    assert [c.symbol for c in out] == ["AAA", "BBB"]
    assert out[0].hv_rank == 0.9


def test_rank_tiebreak_on_dollar_volume() -> None:
    metrics = {
        "HI": ScreenMetrics(dollar_volume=20_000_000.0, hv_rank=0.7),
        "LO": ScreenMetrics(dollar_volume=6_000_000.0, hv_rank=0.7),
    }
    out = rank_candidates(metrics, min_dollar_volume=5_000_000.0)
    assert [c.symbol for c in out] == ["HI", "LO"]


def test_rank_empty_when_all_gated() -> None:
    metrics = {"X": ScreenMetrics(dollar_volume=1.0, hv_rank=0.9)}
    assert rank_candidates(metrics, min_dollar_volume=5_000_000.0) == ()


def test_screen_metrics_from_bars() -> None:
    rng = np.random.default_rng(3)
    n = 120
    closes = 100.0 * np.cumprod(1 + rng.normal(0, 0.02, n))
    bars = pd.DataFrame({"close": closes, "volume": np.full(n, 1_000_000.0)})
    m = screen_metrics(bars)
    assert m is not None
    assert m.dollar_volume > 0
    assert m.hv_rank is not None and 0.0 <= m.hv_rank <= 1.0


def test_screen_metrics_short_history_hv_rank_none() -> None:
    bars = pd.DataFrame({"close": [100.0, 101.0, 102.0], "volume": [1e6, 1e6, 1e6]})
    m = screen_metrics(bars)
    assert m is not None
    assert m.hv_rank is None  # not enough history for an HV series
    assert m.dollar_volume > 0


def test_screen_metrics_missing_columns_returns_none() -> None:
    assert screen_metrics(pd.DataFrame()) is None
    assert screen_metrics(pd.DataFrame({"close": [1.0]})) is None  # no volume
