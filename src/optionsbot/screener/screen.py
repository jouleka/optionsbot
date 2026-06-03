"""Stage-1 universe screener: cheap, chain-free ranking (IBK-95).

Ranks symbols by realized-vol rank (HV-rank, reusing IBK-94) gated by a
liquidity floor (trailing average daily dollar volume). No option chains, no
DB writes. The pure ``rank_candidates`` / ``screen_metrics`` are unit-tested;
``screen_universe`` is the thin I/O orchestrator over a HistoryClient.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from optionsbot.analysis.iv_rank import iv_rank
from optionsbot.analysis.volatility import historical_volatility_series

if TYPE_CHECKING:
    from optionsbot.ibkr.history import HistoryClient
    from optionsbot.scan.types import ScanResult

log = logging.getLogger(__name__)

_VOL_WINDOW = 20


@dataclass(frozen=True, slots=True)
class ScreenMetrics:
    dollar_volume: float
    hv_rank: float | None


@dataclass(frozen=True, slots=True)
class ScreenCandidate:
    symbol: str
    hv_rank: float
    dollar_volume: float


def screen_metrics(bars: pd.DataFrame, vol_window: int = _VOL_WINDOW) -> ScreenMetrics | None:
    """Cheap per-symbol metrics from daily bars. None if bars lack close+volume."""
    if bars.empty or "close" not in bars.columns or "volume" not in bars.columns:
        return None
    recent = bars.tail(vol_window)
    dollar_volume = float((recent["close"] * recent["volume"]).mean())
    hv_series = historical_volatility_series(bars["close"]).dropna()
    if hv_series.empty:
        return ScreenMetrics(dollar_volume=dollar_volume, hv_rank=None)
    hvr = iv_rank(float(hv_series.iloc[-1]), hv_series)
    return ScreenMetrics(dollar_volume=dollar_volume, hv_rank=hvr.rank)


def rank_candidates(
    metrics: dict[str, ScreenMetrics], min_dollar_volume: float
) -> tuple[ScreenCandidate, ...]:
    """Gate by liquidity + a computable HV-rank; sort by hv_rank desc, $vol desc."""
    cands = [
        ScreenCandidate(symbol=sym, hv_rank=m.hv_rank, dollar_volume=m.dollar_volume)
        for sym, m in metrics.items()
        if m.hv_rank is not None and m.dollar_volume >= min_dollar_volume
    ]
    cands.sort(key=lambda c: (c.hv_rank, c.dollar_volume), reverse=True)
    return tuple(cands)


async def screen_universe(
    history_client: HistoryClient,
    universe: Sequence[str],
    min_dollar_volume: float,
    days: int = 252,
) -> tuple[ScreenCandidate, ...]:
    """Fetch cached daily history per symbol, compute metrics, rank.

    Per-symbol failures are logged and skipped so one bad ticker can't abort
    the whole screen.
    """
    metrics: dict[str, ScreenMetrics] = {}
    for symbol in universe:
        try:
            bars = await history_client.get_history(symbol, days=days)
        except Exception:  # noqa: BLE001 -- heterogeneous per-symbol failures
            log.exception("screen: history fetch failed for %s", symbol)
            continue
        m = screen_metrics(bars)
        if m is not None:
            metrics[symbol] = m
    return rank_candidates(metrics, min_dollar_volume)


async def screen_and_scan(
    history_client: HistoryClient,
    universe: Sequence[str],
    min_dollar_volume: float,
    scan_top_n: int,
    scan_one: Callable[[str], Awaitable[ScanResult]],
) -> list[tuple[ScreenCandidate, ScanResult]]:
    """Screen the universe, then full-scan the top-N candidates via ``scan_one``.

    ``scan_one`` is injected (the CLI binds it to ``scan_symbol``) so this stays
    free of IBKR/DB deps and is unit-testable. Per-symbol scan failures are
    logged and skipped so one bad symbol can't abort the batch (mirrors
    ``screen_universe``).
    """
    candidates = await screen_universe(history_client, universe, min_dollar_volume)
    out: list[tuple[ScreenCandidate, ScanResult]] = []
    for cand in candidates[:scan_top_n]:
        try:
            result = await scan_one(cand.symbol)
        except Exception:  # noqa: BLE001 -- per-symbol scan failures are heterogeneous
            log.exception("screen --scan: scan failed for %s", cand.symbol)
            continue
        out.append((cand, result))
    return out
