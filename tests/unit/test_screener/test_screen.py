"""Pure screener ranking + metrics (IBK-95)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from optionsbot.screener.screen import (
    ScreenCandidate,
    ScreenMetrics,
    rank_candidates,
    screen_and_scan,
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


def test_screener_scan_top_n_config(monkeypatch) -> None:
    from optionsbot.config import Settings

    assert Settings().screener.scan_top_n == 5
    monkeypatch.setenv("OPTIONSBOT_SCREENER__SCAN_TOP_N", "8")
    assert Settings().screener.scan_top_n == 8


async def test_screen_and_scan_scans_top_n_and_skips_failures(monkeypatch) -> None:
    cands = (
        ScreenCandidate(symbol="SPY", hv_rank=0.8, dollar_volume=1e9),
        ScreenCandidate(symbol="AAPL", hv_rank=0.7, dollar_volume=5e8),
        ScreenCandidate(symbol="XYZ", hv_rank=0.6, dollar_volume=1e8),
    )

    async def fake_screen_universe(hc, uni, mdv):
        return cands

    monkeypatch.setattr(
        "optionsbot.screener.screen.screen_universe", fake_screen_universe
    )

    calls: list[str] = []

    async def fake_scan_one(symbol: str):
        calls.append(symbol)
        if symbol == "AAPL":
            raise RuntimeError("boom")
        return SimpleNamespace(symbol=symbol, scored=())

    out = await screen_and_scan(
        history_client=object(), universe=[], min_dollar_volume=0.0,
        scan_top_n=2, scan_one=fake_scan_one,
    )
    # top-2 scanned (SPY, AAPL); AAPL raised -> skipped; SPY paired with its candidate.
    assert calls == ["SPY", "AAPL"]
    assert [c.symbol for c, r in out] == ["SPY"]
    assert out[0][1].symbol == "SPY"
