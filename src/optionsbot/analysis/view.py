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
from optionsbot.analysis.types import IVRegime, MarketView

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
    earnings = next_earnings(symbol, manual_overrides=manual_earnings_overrides)
    earnings_in_window = False
    if earnings.next_date is not None:
        today = date.today()
        delta = (earnings.next_date - today).days
        earnings_in_window = 0 <= delta <= earnings_window_days
    return MarketView(
        direction=tr.direction,
        direction_strength=tr.strength,
        iv_regime=_iv_regime_from_rank(ivr.rank),
        iv_rank_value=ivr.rank,
        earnings_in_window=earnings_in_window,
        warming_up=ivr.warming_up,
    )
