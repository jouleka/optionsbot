"""Market-view synthesis.

Combines trend regime, IV rank, and earnings-window detection into a
single ``MarketView`` consumed by strategy scoring (IBK-4 / IBK-5).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from optionsbot.analysis.events import next_earnings
from optionsbot.analysis.iv_rank import iv_rank
from optionsbot.analysis.technicals import trend_regime
from optionsbot.analysis.types import IVRankResult, IVRegime, MarketView
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
    symbol: str,
    bars: pd.DataFrame,
    current_atm_iv: float,
    atm_iv_history: pd.Series,
    earnings_window_days: int = 14,
    manual_earnings_overrides: dict[str, date] | None = None,
) -> MarketView:
    """Synthesize a MarketView from already-fetched data.

    Callers (the daemon scan loop) gather the OHLCV bars and ATM IV
    history from the IBKR layer, then pass them in. This keeps the
    analysis layer free of I/O.
    """
    tr = trend_regime(bars)
    ivr = iv_rank(current_atm_iv, atm_iv_history)
    rank, iv_rank_is_proxy = _resolve_iv_rank(ivr, bars)
    earnings = next_earnings(symbol, manual_overrides=manual_earnings_overrides)
    earnings_in_window = False
    if earnings.next_date is not None:
        today = date.today()
        delta = (earnings.next_date - today).days
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
