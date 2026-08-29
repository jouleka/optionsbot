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


def test_orb_reconciliation_uses_stop_target_ev_when_terminal_ev_turns_negative() -> None:
    """Regression for the live QQQ candidate missed on 2026-08-06."""
    economics = reconcile_entry_economics(
        [
            {
                "symbol": "QQQ",
                "side": "sell",
                "sec_type": "OPT",
                "expiry": "20260806",
                "strike": 720.0,
                "right": "C",
                "quantity": 1,
            },
            {
                "symbol": "QQQ",
                "side": "buy",
                "sec_type": "OPT",
                "expiry": "20260806",
                "strike": 715.0,
                "right": "C",
                "quantity": 1,
            },
        ],
        {
            "credit_or_debit": -173.0,
            "expected_value": 17.25052816113675,
            "terminal_expected_value": 83.03360198143658,
            "prob_profit": 0.564424975189262,
            "managed_target_hit_probability_lcb": 0.564424975189262,
            "opening_range_fvg": {
                "status": "entry_confirmed",
                "source": "trusted_daemon",
                "stop_pct": 0.15,
                "target_r": 1.5,
                "target_pct": 0.225,
            },
        },
        fresh_net_per_share=-2.70,
    )

    assert economics is not None
    assert economics.terminal_expected_value == pytest.approx(-13.9663980185634)
    assert economics.managed_expected_value == pytest.approx(16.64802873791278)
    assert economics.expected_value == pytest.approx(16.64802873791278)

    repriced = reconcile_entry_economics(
        [
            {
                "symbol": "QQQ",
                "side": "sell",
                "sec_type": "OPT",
                "expiry": "20260806",
                "strike": 720.0,
                "right": "C",
                "quantity": 1,
            },
            {
                "symbol": "QQQ",
                "side": "buy",
                "sec_type": "OPT",
                "expiry": "20260806",
                "strike": 715.0,
                "right": "C",
                "quantity": 1,
            },
        ],
        {
            "credit_or_debit": economics.credit_or_debit,
            "expected_value": economics.expected_value,
            "terminal_expected_value": economics.terminal_expected_value,
            "prob_profit": 0.564424975189262,
            "managed_target_hit_probability_lcb": 0.564424975189262,
            "opening_range_fvg": {
                "status": "entry_confirmed",
                "source": "trusted_daemon",
                "stop_pct": 0.15,
                "target_r": 1.5,
                "target_pct": 0.225,
            },
        },
        fresh_net_per_share=-2.71,
    )
    assert repriced is not None
    assert repriced.terminal_expected_value == pytest.approx(-14.9663980185634)
    assert repriced.expected_value is not None and repriced.expected_value > 0


def test_orb_reconciliation_still_rejects_negative_managed_edge() -> None:
    economics = reconcile_entry_economics(
        [_leg("sell", 689.0), _leg("buy", 692.0)],  # type: ignore[list-item]
        {
            "credit_or_debit": -100.0,
            "expected_value": -1.0,
            "terminal_expected_value": 5.0,
            "prob_profit": 0.35,
            "managed_target_hit_probability_lcb": 0.35,
            "opening_range_fvg": {
                "status": "entry_confirmed",
                "source": "trusted_daemon",
                "stop_pct": 0.15,
                "target_r": 1.5,
                "target_pct": 0.225,
            },
        },
        fresh_net_per_share=-1.00,
    )

    assert economics is not None
    assert economics.managed_expected_value == pytest.approx(-1.875)
    assert economics.expected_value == pytest.approx(-1.875)


def test_googl_fresh_economics_deducts_round_trip_costs() -> None:
    economics = reconcile_entry_economics(
        [
            {
                "symbol": "GOOGL", "side": "sell", "sec_type": "OPT",
                "expiry": "20260810", "strike": 350.0, "right": "P", "quantity": 1,
            },
            {
                "symbol": "GOOGL", "side": "buy", "sec_type": "OPT",
                "expiry": "20260810", "strike": 352.5, "right": "P", "quantity": 1,
            },
        ],
        {
            "credit_or_debit": -69.5,
            "terminal_expected_value": 15.524697215831935,
            "prob_profit": 0.34827346286975674,
            "managed_target_hit_probability_lcb": 0.34827346286975674,
            "opening_range_fvg": {
                "status": "entry_confirmed", "source": "trusted_daemon",
                "stop_pct": 0.15, "target_r": 2.0, "target_pct": 0.30,
            },
        },
        fresh_net_per_share=-0.695,
        estimated_round_trip_cost=11.80,
    )

    assert economics is not None
    assert economics.gross_managed_expected_value == pytest.approx(0.4672525512516428)
    assert economics.estimated_round_trip_cost == pytest.approx(11.80)
    assert economics.managed_expected_value == pytest.approx(-11.332747448748358)
    assert economics.expected_value == pytest.approx(-11.332747448748358)


def test_terminal_probability_cannot_authorize_managed_trade() -> None:
    economics = reconcile_entry_economics(
        [_leg("sell", 689.0), _leg("buy", 692.0)],  # type: ignore[list-item]
        {
            "credit_or_debit": -100.0,
            "terminal_expected_value": 50.0,
            "prob_profit": 0.99,
            "opening_range_fvg": {
                "status": "entry_confirmed", "source": "trusted_daemon",
                "stop_pct": 0.15, "target_r": 1.5, "target_pct": 0.225,
            },
        },
        fresh_net_per_share=-1.00,
    )

    assert economics is not None
    assert economics.managed_expected_value is None
    assert economics.expected_value is None


def test_impossible_finite_spread_target_is_unavailable() -> None:
    economics = reconcile_entry_economics(
        [
            _leg("sell", 690.0),
            _leg("buy", 691.0),
        ],  # type: ignore[list-item]
        {
            "credit_or_debit": -82.0,
            "terminal_expected_value": 10.0,
            "prob_profit": 0.70,
            "managed_target_hit_probability_lcb": 0.70,
            "opening_range_fvg": {
                "status": "entry_confirmed", "source": "trusted_daemon",
                "stop_pct": 0.15, "target_r": 1.5, "target_pct": 0.225,
            },
        },
        fresh_net_per_share=-0.82,
        estimated_round_trip_cost=6.80,
    )

    assert economics is not None
    assert economics.max_profit == pytest.approx(18.0)
    assert economics.managed_expected_value is None
    assert economics.expected_value is None
