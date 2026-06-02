"""Tests for the expiry-payoff + probability module (IBK-93)."""

from __future__ import annotations

import math

from optionsbot.scoring.payoff import (
    is_terminal_modelable,
    prob_of_profit,
    terminal_pnl_dollars,
)
from optionsbot.strategies.base import Leg


def _opt(side: str, right: str, strike: float, expiry: str = "20260717") -> Leg:
    return Leg(symbol="SPY", side=side, sec_type="OPT", expiry=expiry, strike=strike, right=right)


def test_terminal_pnl_long_call_below_strike_is_negative_debit() -> None:
    # Long 1 call @100 for $5.00 debit (credit_or_debit = -500). Below strike -> lose the debit.
    legs = (_opt("buy", "C", 100.0),)
    pnl = terminal_pnl_dollars(legs, credit_or_debit=-500.0, s_t=90.0)
    assert pnl == -500.0


def test_terminal_pnl_long_call_breakeven() -> None:
    # Breakeven at strike + debit/100 = 105. P&L ~ 0 there.
    legs = (_opt("buy", "C", 100.0),)
    pnl = terminal_pnl_dollars(legs, credit_or_debit=-500.0, s_t=105.0)
    assert abs(pnl) < 1e-6


def test_terminal_pnl_long_call_above_breakeven_is_positive() -> None:
    legs = (_opt("buy", "C", 100.0),)
    assert terminal_pnl_dollars(legs, credit_or_debit=-500.0, s_t=120.0) == 1500.0


def test_terminal_pnl_long_put_payoff() -> None:
    # Long 1 put @100 for $4.00 debit (credit_or_debit=-400); breakeven at 96.
    legs = (_opt("buy", "P", 100.0),)
    assert terminal_pnl_dollars(legs, credit_or_debit=-400.0, s_t=110.0) == -400.0
    assert abs(terminal_pnl_dollars(legs, credit_or_debit=-400.0, s_t=96.0)) < 1e-6
    assert terminal_pnl_dollars(legs, credit_or_debit=-400.0, s_t=80.0) == 1600.0


def test_is_terminal_modelable_rejects_stock_and_multi_expiry() -> None:
    opt = _opt("buy", "C", 100.0)
    stock = Leg(symbol="SPY", side="buy", sec_type="STK")
    assert is_terminal_modelable((opt,)) is True
    assert is_terminal_modelable((opt, stock)) is False  # stock leg
    # 2 expiries
    assert is_terminal_modelable((opt, _opt("sell", "C", 110.0, expiry="20260821"))) is False


def test_prob_of_profit_none_when_inputs_missing() -> None:
    legs = (_opt("buy", "C", 100.0),)
    assert prob_of_profit(legs, -500.0, spot=100.0, atm_iv=None, dte_days=30) is None
    assert prob_of_profit(legs, -500.0, spot=100.0, atm_iv=0.0, dte_days=30) is None
    assert prob_of_profit(legs, -500.0, spot=100.0, atm_iv=0.2, dte_days=0) is None
    stock = Leg(symbol="SPY", side="buy", sec_type="STK")
    assert prob_of_profit((stock,), -100.0, spot=100.0, atm_iv=0.2, dte_days=30) is None


def test_prob_of_profit_iron_condor_high_for_wide_wings() -> None:
    # Sell 110C / buy 115C / sell 90P / buy 85P, net credit.
    # ~+-10% wings vs ~5.7% sigma -> high prob.
    legs = (
        _opt("sell", "C", 110.0), _opt("buy", "C", 115.0),
        _opt("sell", "P", 90.0), _opt("buy", "P", 85.0),
    )
    p = prob_of_profit(legs, credit_or_debit=120.0, spot=100.0, atm_iv=0.20, dte_days=30)
    assert p is not None
    assert 0.70 < p < 1.0


def test_prob_of_profit_long_otm_call_is_low() -> None:
    # Buy 110C (OTM) for $1.50; needs a big up-move -> low probability.
    legs = (_opt("buy", "C", 110.0),)
    p = prob_of_profit(legs, credit_or_debit=-150.0, spot=100.0, atm_iv=0.20, dte_days=30)
    assert p is not None
    assert 0.0 < p < 0.35


def test_prob_of_profit_is_a_probability() -> None:
    legs = (_opt("buy", "C", 100.0),)
    p = prob_of_profit(legs, -500.0, spot=100.0, atm_iv=0.25, dte_days=45)
    assert p is not None and 0.0 <= p <= 1.0
    assert not math.isnan(p)
