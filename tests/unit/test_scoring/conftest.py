"""Shared fixtures for scoring tests.

The synthetic chain spans strikes 385..415 in $5 increments at ~45 DTE,
with deltas ramping symmetrically so the +/-0.16-delta short legs of an
Iron Condor land safely inside the chain with adjacent strikes available
for wings.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from optionsbot.analysis.types import Direction, IVRegime, MarketView, Strength
from optionsbot.ibkr.types import OptionChainLeg, OptionRight
from optionsbot.strategies import StrategySnapshot, StrategySuggestion
from optionsbot.strategies.iron_condor import IronCondor


def make_view(
    direction: Direction = "neutral",
    strength: Strength = "weak",
    iv_regime: IVRegime = "high",
    earnings_in_window: bool = False,
    iv_rank: float = 0.7,
) -> MarketView:
    return MarketView(
        direction=direction,
        direction_strength=strength,
        iv_regime=iv_regime,
        iv_rank_value=iv_rank,
        earnings_in_window=earnings_in_window,
        warming_up=False,
    )


def _chain_leg(
    expiry: str,
    strike: float,
    right: OptionRight,
    bid: float = 2.0,
    ask: float = 2.2,
    delta: float = 0.16,
    oi: int = 1000,
) -> OptionChainLeg:
    return OptionChainLeg(
        symbol="SPY",
        expiry=expiry,
        strike=strike,
        right=right,
        bid=bid,
        ask=ask,
        iv=0.20,
        delta=delta,
        gamma=0.01,
        theta=-0.02,
        vega=0.1,
        open_interest=oi,
        volume=50,
    )


# Symmetric delta gradient around spot=400 for strikes 385..415 in $5
# increments. The IronCondor short legs at +/-0.16 delta land on K=410 (call)
# and K=390 (put), leaving K=415 / K=385 available as the long-wing strikes.
_STRIKES: tuple[float, ...] = (385.0, 390.0, 395.0, 400.0, 405.0, 410.0, 415.0)
_CALL_DELTAS: dict[float, float] = {
    385.0: 0.85,
    390.0: 0.70,
    395.0: 0.50,
    400.0: 0.35,
    405.0: 0.25,
    410.0: 0.16,
    415.0: 0.08,
}
_PUT_DELTAS: dict[float, float] = {
    385.0: -0.08,
    390.0: -0.16,
    395.0: -0.25,
    400.0: -0.35,
    405.0: -0.50,
    410.0: -0.70,
    415.0: -0.85,
}


@pytest.fixture()
def base_snapshot() -> StrategySnapshot:
    expiry = (date.today() + timedelta(days=45)).strftime("%Y%m%d")
    legs: list[OptionChainLeg] = []
    for k in _STRIKES:
        legs.append(
            _chain_leg(expiry, k, "C", bid=2.0, ask=2.2, delta=_CALL_DELTAS[k])
        )
        legs.append(
            _chain_leg(expiry, k, "P", bid=2.0, ask=2.2, delta=_PUT_DELTAS[k])
        )
    return StrategySnapshot(
        symbol="SPY",
        spot=400.0,
        atm_iv=0.25,
        hv20=0.20,
        iv_rank=0.75,
        chain=tuple(legs),
        view=make_view(),
        dte_target=45,
        position=None,
    )


@pytest.fixture()
def base_suggestion(base_snapshot: StrategySnapshot) -> StrategySuggestion:
    ic = IronCondor()
    s = ic.build_suggestion(base_snapshot, account_value=100_000)
    assert s is not None
    return s


@pytest.fixture()
def base_strategy() -> IronCondor:
    return IronCondor()
