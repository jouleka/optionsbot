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


def test_bull_put_spread() -> None:
    # sell higher-strike put, buy lower-strike put (credit).
    legs = [_opt(95.0, "P", -1.0), _opt(90.0, "P", 1.0)]
    assert identify_structure(legs) == "Bull Put Spread"


def test_bear_put_spread() -> None:
    # buy higher-strike put, sell lower-strike put (debit).
    legs = [_opt(95.0, "P", 1.0), _opt(90.0, "P", -1.0)]
    assert identify_structure(legs) == "Bear Put Spread"


def test_bear_call_spread() -> None:
    # sell lower-strike call, buy higher-strike call (credit).
    legs = [_opt(100.0, "C", -1.0), _opt(105.0, "C", 1.0)]
    assert identify_structure(legs) == "Bear Call Spread"


def test_bull_call_spread() -> None:
    # buy lower-strike call, sell higher-strike call (debit).
    legs = [_opt(100.0, "C", 1.0), _opt(105.0, "C", -1.0)]
    assert identify_structure(legs) == "Bull Call Spread"


def test_vertical_multiple() -> None:
    legs = [_opt(95.0, "P", -2.0), _opt(90.0, "P", 2.0)]
    assert identify_structure(legs) == "Bull Put Spread ×2"


def test_same_strike_two_leg_same_right_is_custom() -> None:
    legs = [_opt(95.0, "P", -1.0), _opt(95.0, "P", 1.0)]  # not a vertical
    assert identify_structure(legs) == "custom (2 legs)"


def test_same_sign_two_leg_same_right_is_custom() -> None:
    legs = [_opt(95.0, "P", 1.0), _opt(90.0, "P", 1.0)]  # two longs -> custom
    assert identify_structure(legs) == "custom (2 legs)"


def test_long_straddle_and_strangle() -> None:
    straddle = [_opt(100.0, "C", 1.0), _opt(100.0, "P", 1.0)]
    strangle = [_opt(105.0, "C", 1.0), _opt(95.0, "P", 1.0)]
    assert identify_structure(straddle) == "Long Straddle"
    assert identify_structure(strangle) == "Long Strangle"


def test_short_straddle_and_strangle() -> None:
    straddle = [_opt(100.0, "C", -1.0), _opt(100.0, "P", -1.0)]
    strangle = [_opt(105.0, "C", -1.0), _opt(95.0, "P", -1.0)]
    assert identify_structure(straddle) == "Short Straddle"
    assert identify_structure(strangle) == "Short Strangle"


def test_synthetic_opposite_sign_opposite_right_is_custom() -> None:
    legs = [_opt(100.0, "C", 1.0), _opt(100.0, "P", -1.0)]  # long call + short put
    assert identify_structure(legs) == "custom (2 legs)"


def test_calendar_and_diagonal() -> None:
    cal = [_opt(100.0, "C", -1.0, expiry="20260717"),
           _opt(100.0, "C", 1.0, expiry="20260821")]
    diag = [_opt(100.0, "C", -1.0, expiry="20260717"),
            _opt(105.0, "C", 1.0, expiry="20260821")]
    assert identify_structure(cal) == "Calendar Spread"
    assert identify_structure(diag) == "Diagonal Spread"


def test_diff_expiry_same_sign_is_custom() -> None:
    legs = [_opt(100.0, "C", 1.0, expiry="20260717"),
            _opt(100.0, "C", 1.0, expiry="20260821")]
    assert identify_structure(legs) == "custom (2 legs)"


def test_iron_condor() -> None:
    legs = [
        _opt(85.0, "P", 1.0),    # long put (lower wing)
        _opt(90.0, "P", -1.0),   # short put
        _opt(110.0, "C", -1.0),  # short call
        _opt(115.0, "C", 1.0),   # long call (upper wing)
    ]
    assert identify_structure(legs) == "Iron Condor"


def test_iron_butterfly() -> None:
    legs = [
        _opt(90.0, "P", 1.0),
        _opt(100.0, "P", -1.0),  # short put at 100
        _opt(100.0, "C", -1.0),  # short call at 100 (same strike -> butterfly)
        _opt(110.0, "C", 1.0),
    ]
    assert identify_structure(legs) == "Iron Butterfly"


def test_iron_condor_multiple() -> None:
    legs = [
        _opt(85.0, "P", 2.0), _opt(90.0, "P", -2.0),
        _opt(110.0, "C", -2.0), _opt(115.0, "C", 2.0),
    ]
    assert identify_structure(legs) == "Iron Condor ×2"


def test_reverse_iron_condor_is_custom() -> None:
    # long the body / short the wings -> sign pattern fails -> custom.
    legs = [
        _opt(85.0, "P", -1.0), _opt(90.0, "P", 1.0),
        _opt(110.0, "C", 1.0), _opt(115.0, "C", -1.0),
    ]
    assert identify_structure(legs) == "custom (4 legs)"


def test_four_legs_three_puts_is_custom() -> None:
    legs = [
        _opt(85.0, "P", 1.0), _opt(90.0, "P", -1.0),
        _opt(95.0, "P", 1.0), _opt(110.0, "C", -1.0),
    ]
    assert identify_structure(legs) == "custom (4 legs)"


def test_iron_condor_mixed_expiry_is_custom() -> None:
    legs = [
        _opt(85.0, "P", 1.0, expiry="20260717"),
        _opt(90.0, "P", -1.0, expiry="20260717"),
        _opt(110.0, "C", -1.0, expiry="20260821"),
        _opt(115.0, "C", 1.0, expiry="20260821"),
    ]
    assert identify_structure(legs) == "custom (4 legs)"


def test_covered_call() -> None:
    legs = [_stk(100.0), _opt(105.0, "C", -1.0)]
    assert identify_structure(legs) == "Covered Call"


def test_covered_call_multiple() -> None:
    legs = [_stk(200.0), _opt(105.0, "C", -2.0)]
    assert identify_structure(legs) == "Covered Call ×2"


def test_covered_call_share_mismatch_is_custom() -> None:
    legs = [_stk(150.0), _opt(105.0, "C", -1.0)]  # 150 != 100*1
    assert identify_structure(legs) == "custom (2 legs)"


def test_stock_plus_long_call_is_custom() -> None:
    legs = [_stk(100.0), _opt(105.0, "C", 1.0)]  # not a covered call
    assert identify_structure(legs) == "custom (2 legs)"


def test_collar_is_custom() -> None:
    legs = [_stk(100.0), _opt(95.0, "P", 1.0), _opt(105.0, "C", -1.0)]
    assert identify_structure(legs) == "custom (3 legs)"


def test_fractional_shares_covered_call_is_custom() -> None:
    # 100.4 shares is NOT 100 -- must not truncate into a "Covered Call" (Opus review S1).
    legs = [_stk(100.4), _opt(105.0, "C", -1.0)]
    assert identify_structure(legs) == "custom (2 legs)"


def test_fractional_option_quantity_is_custom() -> None:
    # A 1.5:1.5 "vertical" isn't a clean whole-contract structure -> custom, not Bull Put.
    legs = [_opt(95.0, "P", -1.5), _opt(90.0, "P", 1.5)]
    assert identify_structure(legs) == "custom (2 legs)"


def test_fractional_shares_stock_only_still_long_stock() -> None:
    # Fractional shares with no options is still unambiguously long stock (no mislabel risk).
    assert identify_structure([_stk(100.5)]) == "Long Stock"
