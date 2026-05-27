"""Output dataclass for the symbol-scan pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from optionsbot.analysis.types import MarketView
from optionsbot.scoring import ScoredStrategy


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Result of scanning one symbol end-to-end.

    ``snapshot_id`` is the autoincrement PK of the newly-inserted
    ``snapshots`` row. ``scored`` is every applicable strategy with its
    composite score and rationale -- callers can apply ``top_k`` for
    display or persist all rows (we persist all in :func:`scan_symbol`).
    """

    symbol: str
    snapshot_id: int
    snapshot_ts: datetime
    view: MarketView
    scored: tuple[ScoredStrategy, ...]
