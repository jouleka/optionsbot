"""Tests for skew-aware PoP/EV (IBK-111)."""

from __future__ import annotations

from optionsbot.ibkr.types import OptionChainLeg
from optionsbot.scoring.payoff import expected_value_dollars, prob_of_profit
from optionsbot.scoring.skew import (
    VolSmile,
    build_smile,
    expected_value_smile,
    prob_of_profit_smile,
)
from optionsbot.strategies.base import Leg


def _opt(side: str, right: str, strike: float, expiry: str = "20260717") -> Leg:
    return Leg(symbol="SPY", side=side, sec_type="OPT", expiry=expiry, strike=strike, right=right)


def _smile(
    spot: float, puts: list[tuple[float, float]], calls: list[tuple[float, float]]
) -> VolSmile:
    return VolSmile(spot=spot, put_points=tuple(sorted(puts)), call_points=tuple(sorted(calls)))


def _chain_leg(
    right: str, strike: float, iv: float | None, expiry: str = "20260717"
) -> OptionChainLeg:
    return OptionChainLeg(
        symbol="SPY", expiry=expiry, strike=strike, right=right, bid=1.0, ask=1.1,
        iv=iv, delta=None, gamma=None, theta=None, vega=None, open_interest=None, volume=None,
    )


# --- Task 1: VolSmile.iv_at / atm_iv ---------------------------------------


def test_iv_at_wing_convention_and_interpolation() -> None:
    s = _smile(
        100.0,
        puts=[(80.0, 0.40), (90.0, 0.30), (100.0, 0.20)],
        calls=[(100.0, 0.22), (110.0, 0.26)],
    )
    # below spot -> put wing
    assert s.iv_at(90.0) == 0.30
    assert abs(s.iv_at(85.0) - 0.35) < 1e-9          # interp between (80,.40) and (90,.30)
    # at/above spot -> call wing
    assert s.iv_at(100.0) == 0.22
    assert abs(s.iv_at(105.0) - 0.24) < 1e-9          # interp between (100,.22) and (110,.26)


def test_iv_at_flat_extrapolation_past_band() -> None:
    s = _smile(100.0, puts=[(80.0, 0.40), (90.0, 0.30)], calls=[(110.0, 0.26)])
    assert s.iv_at(50.0) == 0.40       # below lowest put strike -> clamp
    assert s.iv_at(130.0) == 0.26      # above highest call strike -> clamp


def test_iv_at_empty_wing_falls_back_then_none() -> None:
    only_calls = _smile(100.0, puts=[], calls=[(100.0, 0.20), (110.0, 0.25)])
    # empty put wing -> fall back to the call wing (clamped to nearest)
    assert only_calls.iv_at(90.0) == 0.20
    empty = _smile(100.0, puts=[], calls=[])
    assert empty.iv_at(100.0) is None
    assert empty.atm_iv() is None


def test_atm_iv_is_iv_at_spot() -> None:
    s = _smile(100.0, puts=[(90.0, 0.30)], calls=[(100.0, 0.21), (110.0, 0.25)])
    assert s.atm_iv() == 0.21


# --- Task 2: build_smile ----------------------------------------------------


def test_build_smile_splits_wings_and_skips_bad_iv() -> None:
    chain = (
        _chain_leg("P", 90.0, 0.30), _chain_leg("P", 80.0, 0.40),
        _chain_leg("C", 110.0, 0.25), _chain_leg("C", 100.0, 0.20),
        _chain_leg("P", 95.0, None),   # skipped: None iv
        _chain_leg("C", 105.0, 0.0),   # skipped: non-positive iv
    )
    smile = build_smile(chain, "20260717", spot=100.0)
    assert smile is not None
    assert smile.put_points == ((80.0, 0.40), (90.0, 0.30))     # sorted by strike
    assert smile.call_points == ((100.0, 0.20), (110.0, 0.25))


def test_build_smile_none_when_no_usable_iv() -> None:
    chain = (_chain_leg("P", 90.0, None), _chain_leg("C", 110.0, 0.0))
    assert build_smile(chain, "20260717", spot=100.0) is None


def test_build_smile_ignores_other_expiries() -> None:
    chain = (
        _chain_leg("C", 100.0, 0.20, expiry="20260717"),
        _chain_leg("C", 100.0, 0.5, expiry="20260821"),
    )
    smile = build_smile(chain, "20260717", spot=100.0)
    assert smile is not None
    assert smile.call_points == ((100.0, 0.20),)


# --- Task 3: prob_of_profit_smile ------------------------------------------


def test_prob_smile_flat_matches_lognormal() -> None:
    # Flat smile (all IV = 0.25) must reproduce payoff.prob_of_profit(atm_iv=0.25).
    v = 0.25
    s = _smile(
        100.0,
        puts=[(80.0, v), (90.0, v), (100.0, v)],
        calls=[(100.0, v), (110.0, v), (120.0, v)],
    )
    legs = (_opt("sell", "P", 95.0), _opt("buy", "P", 90.0))  # bull put spread
    credit = 80.0
    flat = prob_of_profit(legs, credit, 100.0, v, 30.0)
    smile_pop = prob_of_profit_smile(legs, credit, 100.0, s, 30.0)
    assert flat is not None and smile_pop is not None
    assert abs(smile_pop - flat) < 0.02


def test_prob_smile_skew_lowers_pop_for_bull_put() -> None:
    # Downward-sloping put smile (richer OTM puts) puts more mass below -> lower PoP.
    skewed = _smile(
        100.0,
        puts=[(80.0, 0.40), (90.0, 0.30), (95.0, 0.25), (100.0, 0.20)],
        calls=[(100.0, 0.20), (110.0, 0.20), (120.0, 0.20)],
    )
    flat = _smile(
        100.0,
        puts=[(80.0, 0.20), (100.0, 0.20)],
        calls=[(100.0, 0.20), (120.0, 0.20)],
    )
    legs = (_opt("sell", "P", 95.0), _opt("buy", "P", 90.0))
    credit = 80.0
    pop_skew = prob_of_profit_smile(legs, credit, 100.0, skewed, 30.0)
    pop_flat = prob_of_profit_smile(legs, credit, 100.0, flat, 30.0)
    assert pop_skew is not None and pop_flat is not None
    assert pop_skew < pop_flat


def test_prob_smile_none_on_degenerate_or_nonmodelable() -> None:
    s = _smile(100.0, puts=[(90.0, 0.25)], calls=[(100.0, 0.25), (110.0, 0.25)])
    legs = (_opt("sell", "P", 95.0), _opt("buy", "P", 90.0))
    assert prob_of_profit_smile(legs, 80.0, 0.0, s, 30.0) is None     # spot <= 0
    assert prob_of_profit_smile(legs, 80.0, 100.0, s, 0.0) is None    # dte <= 0
    stock = Leg(symbol="SPY", side="buy", sec_type="STK")
    assert prob_of_profit_smile((stock,), 0.0, 100.0, s, 30.0) is None  # non-modelable


# --- Task 4: expected_value_smile ------------------------------------------


def test_ev_smile_flat_matches_lognormal() -> None:
    # Flat smile + realized vol r reproduces expected_value_dollars(vol=r),
    # since sigma_EV(K)=r*iv(K)/atm = r everywhere.
    v, r = 0.25, 0.18
    s = _smile(100.0, puts=[(80.0, v), (100.0, v)], calls=[(100.0, v), (120.0, v)])
    legs = (_opt("sell", "P", 95.0), _opt("buy", "P", 90.0))
    credit = 80.0
    flat = expected_value_dollars(legs, credit, 100.0, r, 30.0)
    ev = expected_value_smile(legs, credit, 100.0, s, r, 30.0)
    assert flat is not None and ev is not None
    assert abs(ev - flat) < max(1.0, 0.02 * abs(flat))


def test_ev_smile_skew_lowers_ev_for_bull_put() -> None:
    r = 0.18
    skewed = _smile(
        100.0,
        puts=[(80.0, 0.40), (90.0, 0.30), (95.0, 0.25), (100.0, 0.20)],
        calls=[(100.0, 0.20), (110.0, 0.20), (120.0, 0.20)],
    )
    flat = _smile(
        100.0,
        puts=[(80.0, 0.20), (100.0, 0.20)],
        calls=[(100.0, 0.20), (120.0, 0.20)],
    )
    legs = (_opt("sell", "P", 95.0), _opt("buy", "P", 90.0))
    credit = 80.0
    ev_skew = expected_value_smile(legs, credit, 100.0, skewed, r, 30.0)
    ev_flat = expected_value_smile(legs, credit, 100.0, flat, r, 30.0)
    assert ev_skew is not None and ev_flat is not None
    assert ev_skew < ev_flat


def test_ev_smile_none_without_realized_vol() -> None:
    s = _smile(100.0, puts=[(90.0, 0.25)], calls=[(100.0, 0.25), (110.0, 0.25)])
    legs = (_opt("sell", "P", 95.0), _opt("buy", "P", 90.0))
    assert expected_value_smile(legs, 80.0, 100.0, s, None, 30.0) is None
    assert expected_value_smile(legs, 80.0, 100.0, s, 0.0, 30.0) is None


# --- Two-sided + profit-tail structure coverage (IBK-111 review S2/S3) ------

_CONDOR = (
    _opt("sell", "C", 110.0), _opt("buy", "C", 115.0),
    _opt("sell", "P", 90.0), _opt("buy", "P", 85.0),
)


def test_prob_smile_iron_condor_flat_matches_lognormal_and_bounded() -> None:
    # Two-sided structure: flat smile must still reproduce the flat lognormal, and
    # PoP must be a valid probability in [0, 1].
    v = 0.20
    s = _smile(
        100.0,
        puts=[(80.0, v), (90.0, v), (100.0, v)],
        calls=[(100.0, v), (110.0, v), (120.0, v)],
    )
    flat = prob_of_profit(_CONDOR, 120.0, 100.0, v, 30.0)
    pop = prob_of_profit_smile(_CONDOR, 120.0, 100.0, s, 30.0)
    assert flat is not None and pop is not None
    assert 0.0 <= pop <= 1.0
    assert abs(pop - flat) < 0.02


def test_prob_smile_put_skew_lowers_iron_condor_pop() -> None:
    # Downside put skew raises P(breach the 90 short put) -> lower condor PoP. Exercises
    # BOTH wings (call wing unchanged, only the put wing carries skew).
    skewed = _smile(
        100.0,
        puts=[(80.0, 0.40), (90.0, 0.30), (100.0, 0.20)],
        calls=[(100.0, 0.20), (110.0, 0.20), (120.0, 0.20)],
    )
    flat = _smile(
        100.0,
        puts=[(80.0, 0.20), (100.0, 0.20)],
        calls=[(100.0, 0.20), (120.0, 0.20)],
    )
    pop_skew = prob_of_profit_smile(_CONDOR, 120.0, 100.0, skewed, 30.0)
    pop_flat = prob_of_profit_smile(_CONDOR, 120.0, 100.0, flat, 30.0)
    assert pop_skew is not None and pop_flat is not None
    assert 0.0 <= pop_skew <= 1.0
    assert pop_skew < pop_flat


def test_prob_smile_debit_call_spread_flat_matches_lognormal() -> None:
    # Long-premium / profit-tail structure (bull call debit spread): flat smile must
    # reproduce the flat lognormal and stay bounded.
    v = 0.20
    s = _smile(
        100.0,
        puts=[(80.0, v), (100.0, v)],
        calls=[(100.0, v), (110.0, v), (120.0, v)],
    )
    legs = (_opt("buy", "C", 100.0), _opt("sell", "C", 110.0))
    debit = -250.0  # net debit (negative credit per convention)
    flat = prob_of_profit(legs, debit, 100.0, v, 30.0)
    pop = prob_of_profit_smile(legs, debit, 100.0, s, 30.0)
    assert flat is not None and pop is not None
    assert 0.0 <= pop <= 1.0
    assert abs(pop - flat) < 0.02
