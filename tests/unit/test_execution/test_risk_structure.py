from __future__ import annotations

import pytest

from optionsbot.execution.risk_structure import (
    structural_max_loss_dollars,
    structural_max_profit_dollars,
)


def _leg(side: str, strike: float, right: str) -> dict[str, object]:
    return {
        "symbol": "SPY",
        "side": side,
        "sec_type": "OPT",
        "expiry": "20260717",
        "strike": strike,
        "right": right,
        "quantity": 1,
    }


def test_structural_max_loss_for_credit_spread() -> None:
    legs = [_leg("sell", 580.0, "P"), _leg("buy", 575.0, "P")]
    assert structural_max_loss_dollars(legs, entry_net_per_share=1.20) == pytest.approx(380.0)
    assert structural_max_profit_dollars(
        legs, entry_net_per_share=1.20
    ) == pytest.approx(120.0)


def test_structural_max_loss_for_iron_condor() -> None:
    legs = [
        _leg("sell", 580.0, "P"),
        _leg("buy", 575.0, "P"),
        _leg("sell", 620.0, "C"),
        _leg("buy", 625.0, "C"),
    ]
    assert structural_max_loss_dollars(legs, entry_net_per_share=1.20) == pytest.approx(380.0)


def test_structural_max_loss_for_long_debit_option() -> None:
    legs = [_leg("buy", 600.0, "C")]
    assert structural_max_loss_dollars(legs, entry_net_per_share=-2.00) == pytest.approx(200.0)
    assert structural_max_profit_dollars(legs, entry_net_per_share=-2.00) is None


def test_structural_max_loss_rejects_unbounded_or_duplicate_structure() -> None:
    naked = [_leg("sell", 600.0, "C")]
    duplicate = [_leg("buy", 600.0, "C"), _leg("buy", 600.0, "C")]
    assert structural_max_loss_dollars(naked, entry_net_per_share=2.00) is None
    assert structural_max_loss_dollars(duplicate, entry_net_per_share=-4.00) is None
