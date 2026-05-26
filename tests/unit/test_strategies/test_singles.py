"""Tests for Long Call (IBK-43) and Long Put (IBK-44).

Single-leg, long-premium, directional strategies. Per strategy:

1. ``applicable_view`` -- ``is_applicable`` returns True for the two
   ``(direction, iv_regime)`` tuples in ``applicable_views`` and False for a
   sample of neighbors. Confirms ``long_premium = True``.
2. ``returns_one_buy_leg_correct_right`` -- exactly one leg, ``side="buy"``,
   with the expected ``right`` ("C" for Long Call, "P" for Long Put). The
   strike sits near ATM (~+/-0.50 delta).
3. ``max_loss_equals_debit`` -- ``estimate_credit`` is negative (debit
   paid), ``estimate_max_loss == -credit``.
4. ``max_profit_none`` -- ``estimate_max_profit`` returns ``None``
   (unbounded upside for a long single).

Plus shared ``factor_weights_sum_to_one`` checks.
"""

from __future__ import annotations

import pytest

from optionsbot.analysis.types import MarketView
from optionsbot.ibkr.types import OptionChainLeg
from optionsbot.strategies.base import StrategySnapshot
from optionsbot.strategies.singles import LongCall, LongPut
from tests.unit.test_strategies.conftest import make_view


def _snapshot(
    chain: tuple[OptionChainLeg, ...], view: MarketView
) -> StrategySnapshot:
    return StrategySnapshot(
        symbol="SPY",
        spot=400.0,
        atm_iv=0.20,
        hv20=0.18,
        iv_rank=view.iv_rank_value or 0.5,
        chain=chain,
        view=view,
    )


# ---------------------------------------------------------------------------
# Long Call (IBK-43)
# ---------------------------------------------------------------------------


def test_long_call_applicable_view() -> None:
    s = LongCall()
    assert s.is_applicable(make_view("bull", "low"))
    assert s.is_applicable(make_view("bull", "neutral"))
    assert not s.is_applicable(make_view("bull", "high"))
    assert not s.is_applicable(make_view("bear", "low"))
    assert not s.is_applicable(make_view("neutral", "low"))
    assert s.long_premium is True


def test_long_call_returns_one_buy_leg_correct_right(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = LongCall()
    legs = s.suggest_legs(_snapshot(chain_45dte, make_view("bull", "low")))
    assert legs is not None
    assert len(legs) == 1
    leg = legs[0]
    assert leg.side == "buy"
    assert leg.right == "C"
    assert leg.sec_type == "OPT"
    assert leg.strike is not None
    # ~+0.50 delta is at or near ATM (spot=400). Chain deltas hit ~0.5 between
    # strikes 395 and 405; the picked strike should be in that neighborhood.
    assert 390.0 <= leg.strike <= 410.0
    assert leg.expiry is not None


def test_long_call_max_loss_equals_debit(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = LongCall()
    snap = _snapshot(chain_45dte, make_view("bull", "low"))
    legs = s.suggest_legs(snap)
    assert legs is not None
    credit = s.estimate_credit(legs, snap)
    assert credit < 0  # long premium = debit paid
    max_loss = s.estimate_max_loss(legs, snap)
    assert max_loss is not None
    assert max_loss > 0
    assert max_loss == pytest.approx(-credit)


def test_long_call_max_profit_none(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = LongCall()
    snap = _snapshot(chain_45dte, make_view("bull", "low"))
    legs = s.suggest_legs(snap)
    assert legs is not None
    assert s.estimate_max_profit(legs, snap) is None


def test_long_call_factor_weights_sum_to_one() -> None:
    assert sum(LongCall().factor_weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Long Put (IBK-44)
# ---------------------------------------------------------------------------


def test_long_put_applicable_view() -> None:
    s = LongPut()
    assert s.is_applicable(make_view("bear", "low"))
    assert s.is_applicable(make_view("bear", "neutral"))
    assert not s.is_applicable(make_view("bear", "high"))
    assert not s.is_applicable(make_view("bull", "low"))
    assert not s.is_applicable(make_view("neutral", "low"))
    assert s.long_premium is True


def test_long_put_returns_one_buy_leg_correct_right(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = LongPut()
    legs = s.suggest_legs(_snapshot(chain_45dte, make_view("bear", "low")))
    assert legs is not None
    assert len(legs) == 1
    leg = legs[0]
    assert leg.side == "buy"
    assert leg.right == "P"
    assert leg.sec_type == "OPT"
    assert leg.strike is not None
    # ~-0.50 delta near ATM (spot=400).
    assert 390.0 <= leg.strike <= 410.0
    assert leg.expiry is not None


def test_long_put_max_loss_equals_debit(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = LongPut()
    snap = _snapshot(chain_45dte, make_view("bear", "low"))
    legs = s.suggest_legs(snap)
    assert legs is not None
    credit = s.estimate_credit(legs, snap)
    assert credit < 0  # long premium = debit paid
    max_loss = s.estimate_max_loss(legs, snap)
    assert max_loss is not None
    assert max_loss > 0
    assert max_loss == pytest.approx(-credit)


def test_long_put_max_profit_none(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    s = LongPut()
    snap = _snapshot(chain_45dte, make_view("bear", "low"))
    legs = s.suggest_legs(snap)
    assert legs is not None
    assert s.estimate_max_profit(legs, snap) is None


def test_long_put_factor_weights_sum_to_one() -> None:
    assert sum(LongPut().factor_weights.values()) == pytest.approx(1.0)
