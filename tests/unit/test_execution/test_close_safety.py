"""Tests for close-safety guards (Phase 0 C2): atomic-combo assertion and
post-close naked-short-leg detection."""

from __future__ import annotations

import pytest

from optionsbot.execution.close_safety import (
    NonAtomicCloseError,
    assert_atomic_close_legs,
    find_naked_short_legs,
)
from optionsbot.ibkr.types import PortfolioPosition


def _entry_legs() -> list[dict[str, object]]:
    return [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260918",
         "strike": 580.0, "right": "P", "quantity": 1},
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260918",
         "strike": 575.0, "right": "P", "quantity": 1},
    ]


def _close_legs() -> list[dict[str, object]]:
    # The flipped inverse of the entry (what stage_close_order produces).
    return [
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260918",
         "strike": 580.0, "right": "P", "quantity": 1},
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260918",
         "strike": 575.0, "right": "P", "quantity": 1},
    ]


def _pos(strike: float, right: str, position: float) -> PortfolioPosition:
    return PortfolioPosition(
        account="DU1", symbol="SPY", sec_type="OPT", expiry="20260918",
        strike=strike, right=right,  # type: ignore[arg-type]
        multiplier=100, position=position, avg_cost=0.0, market_price=None,
        market_value=None, unrealized_pnl=None, realized_pnl=None,
    )


def test_atomic_assert_passes_for_proper_inverse_multileg() -> None:
    # A multi-leg close that is the exact inverse of the entry routes as one
    # atomic BAG — no exception.
    assert_atomic_close_legs(entry_legs=_entry_legs(), close_legs=_close_legs())


def test_atomic_assert_rejects_single_leg_close_of_multileg_entry() -> None:
    # A 2-leg entry whose close has only ONE option leg cannot be closed
    # atomically — fail safe rather than leg out and strand the other side.
    bad_close = _close_legs()[:1]
    with pytest.raises(NonAtomicCloseError):
        assert_atomic_close_legs(entry_legs=_entry_legs(), close_legs=bad_close)


def test_atomic_assert_rejects_close_not_inverse_of_entry() -> None:
    # Close legs that are not the side-flipped inverse of the entry are not a
    # safe atomic close of THIS position.
    wrong = [
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260918",
         "strike": 580.0, "right": "P", "quantity": 1},
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260918",
         "strike": 575.0, "right": "P", "quantity": 1},
    ]
    with pytest.raises(NonAtomicCloseError):
        assert_atomic_close_legs(entry_legs=_entry_legs(), close_legs=wrong)


def test_find_naked_short_legs_flat_after_full_close() -> None:
    # Position fully flat at the broker -> no naked legs.
    positions = [_pos(580.0, "P", 0.0), _pos(575.0, "P", 0.0)]
    assert find_naked_short_legs(_entry_legs(), positions) == []


def test_find_naked_short_legs_detects_residual_short() -> None:
    # The bought (long) leg closed but the SOLD (short) leg is still open at
    # the broker: a residual naked short -> P1.
    positions = [_pos(580.0, "P", -1.0), _pos(575.0, "P", 0.0)]
    naked = find_naked_short_legs(_entry_legs(), positions)
    assert len(naked) == 1
    assert naked[0].strike == 580.0
    assert naked[0].right == "P"


def test_find_naked_short_legs_ignores_residual_long() -> None:
    # A residual LONG leg is not naked-short risk (defined, capital already
    # spent) — only short legs are flagged.
    positions = [_pos(580.0, "P", 0.0), _pos(575.0, "P", 1.0)]
    assert find_naked_short_legs(_entry_legs(), positions) == []
