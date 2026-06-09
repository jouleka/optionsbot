"""Tests for per-structure reconstruction (IBK-120)."""

from __future__ import annotations

from optionsbot.analysis.structures import identify_structure
from optionsbot.ibkr.types import PortfolioPosition


def _opt(
    strike: float, right: str, position: float, expiry: str = "20260717", symbol: str = "SPY"
) -> PortfolioPosition:
    return PortfolioPosition(
        account="DU1", symbol=symbol, sec_type="OPT", expiry=expiry, strike=strike,
        right=right, multiplier=100, position=position, avg_cost=100.0,  # type: ignore[arg-type]
        market_price=1.0, market_value=-100.0, unrealized_pnl=0.0, realized_pnl=0.0,
    )


def _stk(position: float, symbol: str = "SPY") -> PortfolioPosition:
    return PortfolioPosition(
        account="DU1", symbol=symbol, sec_type="STK", expiry=None, strike=None, right=None,
        multiplier=1, position=position, avg_cost=100.0, market_price=100.0,
        market_value=position * 100.0, unrealized_pnl=0.0, realized_pnl=0.0,
    )


def test_single_legs() -> None:
    assert identify_structure([_opt(95.0, "C", 1.0)]) == "Long Call"
    assert identify_structure([_opt(95.0, "C", -1.0)]) == "Short Call"
    assert identify_structure([_opt(95.0, "P", 1.0)]) == "Long Put"
    assert identify_structure([_opt(95.0, "P", -1.0)]) == "Short Put"


def test_single_leg_multiple() -> None:
    assert identify_structure([_opt(95.0, "C", 3.0)]) == "Long Call ×3"


def test_stock_only() -> None:
    assert identify_structure([_stk(100.0)]) == "Long Stock"
    assert identify_structure([_stk(-100.0)]) == "Short Stock"


def test_ratio_is_custom() -> None:
    # short 2 / long 1 put -> uneven |qty| -> not a clean structure.
    legs = [_opt(95.0, "P", -2.0), _opt(90.0, "P", 1.0)]
    assert identify_structure(legs) == "custom (2 legs)"


def test_three_leg_is_custom() -> None:
    legs = [_opt(95.0, "P", -1.0), _opt(90.0, "P", 1.0), _opt(105.0, "C", -1.0)]
    assert identify_structure(legs) == "custom (3 legs)"


def test_zero_position_legs_ignored() -> None:
    legs = [_opt(95.0, "C", 1.0), _opt(90.0, "C", 0.0)]
    assert identify_structure(legs) == "Long Call"
