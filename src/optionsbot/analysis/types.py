"""Dataclasses returned by the analysis layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date_type
from typing import Literal


@dataclass(frozen=True, slots=True)
class EarningsInfo:
    """Next earnings date for a symbol with provenance.

    ``source`` indicates where the date came from:
      * ``"manual"`` -- caller-supplied override
      * ``"yfinance"`` -- looked up via the yfinance package
      * ``"unknown"`` -- no date could be determined (network failure,
        missing calendar, or symbol has no upcoming earnings)
    """

    next_date: _date_type | None
    source: Literal["yfinance", "manual", "unknown"]


@dataclass(frozen=True, slots=True)
class IVRankResult:
    """Result of an IV-rank computation.

    ``warming_up`` is True when fewer than 30 daily IV snapshots are
    available; ``rank`` is still returned (over whatever history exists)
    but callers should treat it as low-confidence.
    """

    rank: float | None  # 0.0 .. 1.0, or None when no history at all
    warming_up: bool
    sample_size: int


Direction = Literal["bull", "neutral", "bear"]
Strength = Literal["strong", "weak"]


@dataclass(frozen=True, slots=True)
class TrendRegime:
    direction: Direction
    strength: Strength
    adx: float | None
    sma20: float | None
    sma50: float | None
