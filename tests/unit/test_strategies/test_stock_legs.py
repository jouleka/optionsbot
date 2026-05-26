"""Tests for Covered Call (IBK-41) and Cash-Secured Put (IBK-42).

Both are short-premium strategies built around existing capital -- either
a 100-share equity position (Covered Call) or sufficient cash to absorb
assignment (Cash-Secured Put). Coverage:

CoveredCall:
1. ``test_covered_call_returns_none_without_position`` -- snapshot.position is None.
2. ``test_covered_call_returns_none_with_insufficient_shares`` -- position < 100.
3. ``test_covered_call_returns_two_legs_with_100_shares`` -- eligible, 2 legs.
4. ``test_covered_call_includes_stock_leg_and_short_call`` -- leg shapes (STK + OPT).
5. ``test_covered_call_max_loss_formula`` -- position * spot - credit.
6. ``test_covered_call_max_profit_formula`` -- position * (call_strike - avg_cost) + credit.
7. ``test_covered_call_applicable_views`` -- bull/high, neutral/high, bull/neutral.
8. ``test_covered_call_factor_weights_sum_to_one``.

CashSecuredPut:
9. ``test_csp_returns_one_short_put`` -- single sell P leg.
10. ``test_csp_applicable_to_bull_high_iv``.
11. ``test_csp_max_loss_formula`` -- put_strike * 100 - credit.
12. ``test_csp_suggest_size_zero_when_account_too_small`` -- the size cap.
13. ``test_csp_factor_weights_sum_to_one``.
"""

from __future__ import annotations

import pytest

from optionsbot.analysis.types import MarketView
from optionsbot.ibkr.types import OptionChainLeg, PositionRecord
from optionsbot.strategies.base import StrategySnapshot
from optionsbot.strategies.stock_legs import CashSecuredPut, CoveredCall
from tests.unit.test_strategies.conftest import make_view


def _position(shares: float = 100.0, avg_cost: float = 395.0) -> PositionRecord:
    return PositionRecord(
        account="DU1234567",
        symbol="SPY",
        sec_type="STK",
        exchange="SMART",
        currency="USD",
        position=shares,
        avg_cost=avg_cost,
    )


def _snapshot(
    chain: tuple[OptionChainLeg, ...],
    view: MarketView,
    position: PositionRecord | None = None,
) -> StrategySnapshot:
    return StrategySnapshot(
        symbol="SPY",
        spot=400.0,
        atm_iv=0.20,
        hv20=0.18,
        iv_rank=0.75,
        chain=chain,
        view=view,
        position=position,
    )


# ---------------------------------------------------------------------------
# Covered Call (IBK-41)
# ---------------------------------------------------------------------------


def test_covered_call_returns_none_without_position(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    cc = CoveredCall()
    snap = _snapshot(chain_45dte, make_view("bull", "high"), position=None)
    assert cc.suggest_legs(snap) is None


def test_covered_call_returns_none_with_insufficient_shares(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    cc = CoveredCall()
    snap = _snapshot(
        chain_45dte, make_view("bull", "high"), position=_position(shares=50.0)
    )
    assert cc.suggest_legs(snap) is None


def test_covered_call_returns_two_legs_with_100_shares(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    cc = CoveredCall()
    snap = _snapshot(
        chain_45dte, make_view("bull", "high"), position=_position(shares=100.0)
    )
    legs = cc.suggest_legs(snap)
    assert legs is not None
    assert len(legs) == 2


def test_covered_call_includes_stock_leg_and_short_call(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    cc = CoveredCall()
    snap = _snapshot(
        chain_45dte, make_view("bull", "high"), position=_position(shares=100.0)
    )
    legs = cc.suggest_legs(snap)
    assert legs is not None
    stock_leg = next((leg for leg in legs if leg.sec_type == "STK"), None)
    option_leg = next((leg for leg in legs if leg.sec_type == "OPT"), None)
    assert stock_leg is not None
    assert option_leg is not None
    # Stock leg represents the existing long position.
    assert stock_leg.side == "buy"
    assert stock_leg.quantity == 100
    assert stock_leg.expiry is None
    assert stock_leg.strike is None
    assert stock_leg.right is None
    # Option leg is a short call at ~+0.30 delta.
    assert option_leg.side == "sell"
    assert option_leg.right == "C"
    assert option_leg.strike is not None
    assert option_leg.expiry is not None


def test_covered_call_max_loss_formula(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    cc = CoveredCall()
    snap = _snapshot(
        chain_45dte, make_view("bull", "high"), position=_position(shares=100.0)
    )
    legs = cc.suggest_legs(snap)
    assert legs is not None
    credit = cc.estimate_credit(legs, snap)
    assert credit > 0  # short call collects premium
    max_loss = cc.estimate_max_loss(legs, snap)
    assert max_loss is not None
    # Worst case: stock to 0. Loss = position * spot - credit.
    expected = snap.position.position * snap.spot - credit  # type: ignore[union-attr]
    assert max_loss == pytest.approx(expected)


def test_covered_call_max_profit_formula(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    cc = CoveredCall()
    snap = _snapshot(
        chain_45dte, make_view("bull", "high"), position=_position(shares=100.0)
    )
    legs = cc.suggest_legs(snap)
    assert legs is not None
    credit = cc.estimate_credit(legs, snap)
    max_profit = cc.estimate_max_profit(legs, snap)
    assert max_profit is not None
    # Capped at assignment: position * (call_strike - avg_cost) + credit.
    call_strike = next(
        leg.strike for leg in legs if leg.sec_type == "OPT" and leg.strike is not None
    )
    expected = (
        snap.position.position * (call_strike - snap.position.avg_cost) + credit  # type: ignore[union-attr]
    )
    assert max_profit == pytest.approx(expected)


def test_covered_call_applicable_views() -> None:
    cc = CoveredCall()
    assert cc.is_applicable(make_view("bull", "high"))
    assert cc.is_applicable(make_view("neutral", "high"))
    assert cc.is_applicable(make_view("bull", "neutral"))
    assert not cc.is_applicable(make_view("bear", "high"))
    assert not cc.is_applicable(make_view("neutral", "low"))
    assert cc.long_premium is False
    assert cc.defined_risk is True


def test_covered_call_factor_weights_sum_to_one() -> None:
    assert sum(CoveredCall().factor_weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cash-Secured Put (IBK-42)
# ---------------------------------------------------------------------------


def test_csp_returns_one_short_put(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    csp = CashSecuredPut()
    snap = _snapshot(chain_45dte, make_view("bull", "high"))
    legs = csp.suggest_legs(snap)
    assert legs is not None
    assert len(legs) == 1
    leg = legs[0]
    assert leg.side == "sell"
    assert leg.right == "P"
    assert leg.sec_type == "OPT"
    assert leg.strike is not None
    assert leg.expiry is not None


def test_csp_applicable_to_bull_high_iv() -> None:
    csp = CashSecuredPut()
    assert csp.is_applicable(make_view("bull", "high"))
    assert csp.is_applicable(make_view("neutral", "high"))
    assert not csp.is_applicable(make_view("bull", "low"))
    assert not csp.is_applicable(make_view("bear", "high"))
    assert csp.long_premium is False
    assert csp.defined_risk is True


def test_csp_max_loss_formula(
    chain_45dte: tuple[OptionChainLeg, ...],
) -> None:
    csp = CashSecuredPut()
    snap = _snapshot(chain_45dte, make_view("bull", "high"))
    legs = csp.suggest_legs(snap)
    assert legs is not None
    credit = csp.estimate_credit(legs, snap)
    assert credit > 0
    put_strike = legs[0].strike
    assert put_strike is not None
    max_loss = csp.estimate_max_loss(legs, snap)
    assert max_loss is not None
    # Worst case: assigned, then stock to 0. Loss = put_strike * 100 - credit.
    assert max_loss == pytest.approx(put_strike * 100 - credit)
    # max_profit = credit (put expires worthless).
    assert csp.estimate_max_profit(legs, snap) == pytest.approx(credit)


def test_csp_suggest_size_zero_when_account_too_small() -> None:
    csp = CashSecuredPut()
    # Suppose max_loss_per_contract = $39_000 (put strike $390 * 100 minus a small credit).
    max_loss = 39_000.0
    # Tiny account — can't even afford one assignment.
    assert csp.suggest_size(account_value=10_000.0, max_loss_per_contract=max_loss) == 0
    # Large enough account: base sizing kicks in.
    assert (
        csp.suggest_size(account_value=5_000_000.0, max_loss_per_contract=max_loss) >= 1
    )


def test_csp_factor_weights_sum_to_one() -> None:
    assert sum(CashSecuredPut().factor_weights.values()) == pytest.approx(1.0)
