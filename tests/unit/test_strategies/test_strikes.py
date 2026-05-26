"""Tests for strike-selection helpers."""

from __future__ import annotations

from datetime import date, timedelta

from optionsbot.ibkr.types import OptionChainLeg, OptionRight
from optionsbot.strategies.strikes import (
    closest_expiry_to_dte,
    closest_strike,
    filter_by_expiry,
    filter_by_right,
    find_strike_by_delta,
)


def _leg(
    expiry: str, strike: float, right: OptionRight, delta: float | None = None
) -> OptionChainLeg:
    return OptionChainLeg(
        symbol="SPY",
        expiry=expiry,
        strike=strike,
        right=right,
        bid=1.0,
        ask=1.1,
        iv=0.2,
        delta=delta,
        gamma=0.01,
        theta=-0.02,
        vega=0.1,
        open_interest=100,
        volume=10,
    )


def test_filter_by_expiry() -> None:
    chain = (_leg("20260619", 400, "C"), _leg("20260717", 400, "C"))
    assert len(filter_by_expiry(chain, "20260619")) == 1


def test_filter_by_right_returns_only_matching() -> None:
    chain = (_leg("20260619", 400, "C"), _leg("20260619", 400, "P"))
    assert all(leg.right == "C" for leg in filter_by_right(chain, "C"))


def test_closest_expiry_to_dte_picks_nearest() -> None:
    today = date.today()
    expiries = [(today + timedelta(days=d)).strftime("%Y%m%d") for d in (10, 35, 70)]
    chain = tuple(_leg(e, 400, "C") for e in expiries)
    assert closest_expiry_to_dte(chain, dte_target=45, today=today) == expiries[1]


def test_closest_expiry_returns_none_for_empty_chain() -> None:
    assert closest_expiry_to_dte((), dte_target=45) is None


def test_closest_strike_picks_nearest() -> None:
    legs = tuple(_leg("20260619", k, "C") for k in (395.0, 400.0, 405.0))
    nearest = closest_strike(legs, 402.0)
    assert nearest is not None
    assert nearest.strike == 400.0


def test_find_strike_by_delta_call() -> None:
    legs = (
        _leg("20260619", 400, "C", delta=0.30),
        _leg("20260619", 410, "C", delta=0.15),
        _leg("20260619", 420, "C", delta=0.05),
    )
    # Target +0.16 -> closest is 0.15 leg (strike 410)
    found = find_strike_by_delta(legs, 0.16, "C")
    assert found is not None
    assert found.strike == 410.0


def test_find_strike_by_delta_put() -> None:
    legs = (
        _leg("20260619", 395, "P", delta=-0.30),
        _leg("20260619", 385, "P", delta=-0.16),
        _leg("20260619", 375, "P", delta=-0.05),
    )
    # Target -0.16 -> closest is the -0.16 leg
    found = find_strike_by_delta(legs, -0.16, "P")
    assert found is not None
    assert found.strike == 385.0


def test_find_strike_by_delta_returns_none_when_no_deltas() -> None:
    legs = (_leg("20260619", 400, "C", delta=None),)
    assert find_strike_by_delta(legs, 0.16, "C") is None
