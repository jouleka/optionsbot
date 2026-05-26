"""IV rank computed from a series of daily ATM IV snapshots.

IBKR does not provide historical IV; the daemon stores daily ATM IV
snapshots and this function turns those into a rank in [0, 1].
"""

from __future__ import annotations

import pandas as pd

from optionsbot.analysis.types import IVRankResult

_DEFAULT_WINDOW = 252  # trading days
_MIN_CONFIDENT_SAMPLES = 30  # below this, flag warming_up


def iv_rank(
    current_iv: float,
    history: pd.Series,
    window: int = _DEFAULT_WINDOW,
) -> IVRankResult:
    """Compute (current - min) / (max - min) over the trailing ``window``.

    Args:
        current_iv: Today's ATM IV.
        history: Historical daily ATM IV snapshots. Most recent last.
        window: Trailing window size. Defaults to 252 trading days.

    Returns:
        IVRankResult with rank in [0, 1] (clamped), warming_up flag,
        and the actual sample size used. ``rank`` is None when history
        is empty or constant (no range to normalise against).
    """
    tail = history.tail(window).dropna()
    n = len(tail)
    if n == 0:
        return IVRankResult(rank=None, warming_up=True, sample_size=0)
    lo = float(tail.min())
    hi = float(tail.max())
    warming_up = n < _MIN_CONFIDENT_SAMPLES
    if hi == lo:
        return IVRankResult(rank=None, warming_up=warming_up, sample_size=n)
    raw = (current_iv - lo) / (hi - lo)
    clamped = max(0.0, min(1.0, raw))
    return IVRankResult(rank=clamped, warming_up=warming_up, sample_size=n)
