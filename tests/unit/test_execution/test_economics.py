from __future__ import annotations

import pytest

from optionsbot.execution.economics import reconcile_entry_economics


def _leg(side: str, strike: float) -> dict[str, object]:
    return {
        "symbol": "QQQ",
        "side": side,
        "sec_type": "OPT",
        "expiry": "20260723",
        "strike": strike,
        "right": "P",
        "quantity": 1,
    }


def test_reconciles_qqq_debit_spread_and_reprices_expected_value() -> None:
    economics = reconcile_entry_economics(
        [_leg("sell", 689.0), _leg("buy", 692.0)],  # type: ignore[list-item]
        {
            "credit_or_debit": -90.50,
            "max_loss": 90.50,
            "max_profit": 209.50,
            "expected_value": 6.329505005969168,
        },
        fresh_net_per_share=-1.23,
    )

    assert economics is not None
    assert economics.credit_or_debit == pytest.approx(-123.0)
    assert economics.max_loss == pytest.approx(123.0)
    assert economics.max_profit == pytest.approx(177.0)
    assert economics.reward_risk == pytest.approx(177.0 / 123.0)
    assert economics.expected_value == pytest.approx(-26.170494994030832)


def test_reconciles_credit_spread() -> None:
    economics = reconcile_entry_economics(
        [_leg("sell", 692.0), _leg("buy", 689.0)],  # type: ignore[list-item]
        {"credit_or_debit": 115.0, "expected_value": 8.0},
        fresh_net_per_share=1.20,
    )

    assert economics is not None
    assert economics.credit_or_debit == pytest.approx(120.0)
    assert economics.max_loss == pytest.approx(180.0)
    assert economics.max_profit == pytest.approx(120.0)
    assert economics.expected_value == pytest.approx(13.0)
