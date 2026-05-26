"""Dataclasses returned by the analysis layer."""

from __future__ import annotations

from dataclasses import dataclass


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
