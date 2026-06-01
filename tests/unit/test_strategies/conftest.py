"""Synthetic options chain fixtures for strategy tests.

Spot is ``400``. The 45-DTE chain covers strikes 360..440 in $5 increments
with deltas ramping symmetrically so the +/-0.16-delta short legs of an
Iron Condor (and similar strategies) land safely inside the chain with
strikes on either side for wings.

Premiums are tied to absolute delta so deep-ITM legs are richer than
deep-OTM legs and short-premium structures generate positive credits.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from optionsbot.analysis.types import Direction, IVRegime, MarketView
from optionsbot.ibkr.types import OptionChainLeg, OptionRight


def _make_leg(
    symbol: str,
    expiry: str,
    strike: float,
    right: OptionRight,
    bid: float,
    ask: float,
    delta: float,
    iv: float = 0.20,
    oi: int = 1000,
    vol: int = 50,
) -> OptionChainLeg:
    return OptionChainLeg(
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        right=right,
        bid=bid,
        ask=ask,
        iv=iv,
        delta=delta,
        gamma=0.01,
        theta=-0.02,
        vega=0.1,
        open_interest=oi,
        volume=vol,
    )


def _price_from_delta(delta: float) -> tuple[float, float]:
    """Quick (bid, ask) pair: pricier when |delta| is larger."""
    bid = max(0.5, abs(delta) * 30.0)
    return bid, bid + 0.10


@pytest.fixture()
def chain_45dte() -> tuple[OptionChainLeg, ...]:
    """SPY-style chain around spot=400 at ~45 DTE.

    Strikes 360..440 in $5 increments. For calls, deltas ramp from ~0.95
    (deep ITM at strike 360) down to ~0.05 (deep OTM at strike 440). For
    puts, the mirror: ~-0.95 at the high strike, ~-0.05 at the low strike.
    """
    expiry = (date.today() + timedelta(days=45)).strftime("%Y%m%d")
    strikes = list(range(360, 445, 5))  # 17 strikes
    legs: list[OptionChainLeg] = []
    # Calls
    for i, k in enumerate(strikes):
        raw_delta = 0.95 - i * 0.06
        delta = max(0.03, min(0.97, raw_delta))
        bid, ask = _price_from_delta(delta)
        legs.append(_make_leg("SPY", expiry, float(k), "C", bid=bid, ask=ask, delta=delta))
    # Puts (mirror: highest strike is deepest ITM, lowest strike is deepest OTM)
    for i, k in enumerate(reversed(strikes)):
        raw_delta = -0.95 + i * 0.06
        delta = min(-0.03, max(-0.97, raw_delta))
        bid, ask = _price_from_delta(delta)
        legs.append(_make_leg("SPY", expiry, float(k), "P", bid=bid, ask=ask, delta=delta))
    return tuple(legs)


@pytest.fixture()
def chain_multi_dte() -> tuple[OptionChainLeg, ...]:
    """Chain at 30, 45, and 60 DTE for calendar/diagonal tests."""
    all_legs: list[OptionChainLeg] = []
    for dte in (30, 45, 60):
        exp = (date.today() + timedelta(days=dte)).strftime("%Y%m%d")
        for k in (395, 400, 405):
            for right, delta in (("C", 0.50), ("P", -0.50)):
                # Longer DTE -> higher premium
                base = 5.0 + dte * 0.05
                all_legs.append(
                    _make_leg(
                        "SPY",
                        exp,
                        float(k),
                        right,  # type: ignore[arg-type]
                        bid=base,
                        ask=base + 0.1,
                        delta=delta,
                    )
                )
    return tuple(all_legs)


@pytest.fixture()
def chain_front_back() -> tuple[OptionChainLeg, ...]:
    """Chain at 45 + 75 DTE -- the scan's front+back fetch (gap 30, the
    minimum a calendar/diagonal needs)."""
    all_legs: list[OptionChainLeg] = []
    for dte in (45, 75):
        exp = (date.today() + timedelta(days=dte)).strftime("%Y%m%d")
        for k in (395, 400, 405):
            for right, delta in (("C", 0.50), ("P", -0.50)):
                base = 5.0 + dte * 0.05
                all_legs.append(
                    _make_leg(
                        "SPY", exp, float(k), right,  # type: ignore[arg-type]
                        bid=base, ask=base + 0.1, delta=delta,
                    )
                )
    return tuple(all_legs)


def make_view(
    direction: Direction = "neutral",
    iv_regime: IVRegime = "high",
) -> MarketView:
    return MarketView(
        direction=direction,
        direction_strength="weak",
        iv_regime=iv_regime,
        iv_rank_value=0.7 if iv_regime == "high" else 0.4,
        earnings_in_window=False,
        warming_up=False,
    )


@pytest.fixture()
def neutral_high_iv_view() -> MarketView:
    return make_view("neutral", "high")


@pytest.fixture()
def bullish_high_iv_view() -> MarketView:
    return make_view("bull", "high")
