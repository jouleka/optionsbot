"""Market-view synthesis.

Combines trend regime, IV rank, and earnings-window detection into a
single ``MarketView`` consumed by strategy scoring (IBK-4 / IBK-5).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from optionsbot.analysis.iv_rank import iv_rank
from optionsbot.analysis.technicals import trend_regime
from optionsbot.analysis.types import EarningsInfo, IVRankResult, IVRegime, MarketView
from optionsbot.analysis.volatility import historical_volatility_series

_IV_LOW_CEILING = 0.30
_IV_HIGH_FLOOR = 0.60


def _iv_regime_from_rank(rank: float | None) -> IVRegime:
    if rank is None:
        return "neutral"
    if rank < _IV_LOW_CEILING:
        return "low"
    if rank >= _IV_HIGH_FLOOR:
        return "high"
    return "neutral"


def _resolve_iv_rank(
    ivr: IVRankResult, bars: pd.DataFrame
) -> tuple[float | None, bool]:
    """Return ``(rank, is_proxy)``.

    While IV history is warming up, rank today's realized vol against its own
    history (the HV-rank proxy) instead of the thin IV sample. Falls back to the
    IV-based rank when IV history is mature or when there's not enough price
    history to build an HV series.
    """
    if not ivr.warming_up:
        return ivr.rank, False
    if "close" not in bars.columns or bars.empty:
        return ivr.rank, False
    valid = historical_volatility_series(bars["close"]).dropna()
    if valid.empty:
        return ivr.rank, False
    hvr = iv_rank(float(valid.iloc[-1]), valid)
    if hvr.rank is None:
        return ivr.rank, False
    return hvr.rank, True


def infer_view(
    bars: pd.DataFrame,
    current_atm_iv: float,
    atm_iv_history: pd.Series,
    earnings: EarningsInfo,
    earnings_window_days: int = 14,
    today: date | None = None,
) -> MarketView:
    """Synthesize a MarketView from already-fetched data.

    Pure + I/O-free: the caller (the daemon scan loop) gathers the OHLCV bars,
    ATM IV history, AND the next-earnings date (the latter fetched off the event
    loop with a timeout -- see scan.symbol), then passes them in. Keeping the
    yfinance lookup out of here is what stops a slow/hung Yahoo response from
    blocking the asyncio loop (IBK-149).
    """
    tr = trend_regime(bars)
    ivr = iv_rank(current_atm_iv, atm_iv_history)
    rank, iv_rank_is_proxy = _resolve_iv_rank(ivr, bars)
    earnings_in_window = False
    if earnings.next_date is not None:
        reference = today if today is not None else date.today()
        delta = (earnings.next_date - reference).days
        earnings_in_window = 0 <= delta <= earnings_window_days
    return MarketView(
        direction=tr.direction,
        direction_strength=tr.strength,
        iv_regime=_iv_regime_from_rank(rank),
        iv_rank_value=rank,
        earnings_in_window=earnings_in_window,
        warming_up=ivr.warming_up,
        iv_rank_is_proxy=iv_rank_is_proxy,
    )
