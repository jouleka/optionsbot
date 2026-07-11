"""Tests for IBK-138 daemon-side request_exit gates."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from optionsbot.execution.exit_requests import (
    ExitRequestGateInput,
    QuoteGateState,
    evaluate_exit_request_gate,
    evaluate_hermes_loss_cap,
)

NOW = datetime(2026, 7, 9, 15, 0, tzinfo=UTC)


def _base(**overrides: object) -> ExitRequestGateInput:
    data: dict[str, object] = {
        "position_id": 7,
        "catalyst_type": "downgrade_upgrade",
        "confidence": 0.82,
        "sources": ["Reuters", "confirmed adverse price/volume"],
        "reason": "material downgrade with adverse tape",
        "today_position_requests": 0,
        "today_portfolio_requests": 0,
    }
    data.update(overrides)
    return ExitRequestGateInput(**data)  # type: ignore[arg-type]


def test_gate_refuses_low_confidence() -> None:
    gate = evaluate_exit_request_gate(
        _base(confidence=0.40),
        QuoteGateState(entry_net=1.20, current_net=1.10, dte=30, deterministic_exit_reason=None),
    )

    assert gate.allowed is False
    assert "confidence" in gate.reason


def test_gate_refuses_winning_headline_close() -> None:
    gate = evaluate_exit_request_gate(
        _base(catalyst_type="headline_news"),
        # Credit spread collected 1.20, now costs 0.50: winner.
        QuoteGateState(entry_net=1.20, current_net=0.50, dte=30, deterministic_exit_reason=None),
    )

    assert gate.allowed is False
    assert "winner" in gate.reason


def test_gate_refuses_deterministic_hold_without_adverse_move() -> None:
    gate = evaluate_exit_request_gate(
        _base(),
        # Losing, but less than the adverse-move threshold, and deterministic exit says HOLD.
        QuoteGateState(entry_net=1.20, current_net=1.35, dte=30, deterministic_exit_reason=None),
    )

    assert gate.allowed is False
    assert "deterministic HOLD" in gate.reason


def test_gate_allows_corroborated_adverse_loser() -> None:
    gate = evaluate_exit_request_gate(
        _base(),
        # Loss is 0.45 on 1.20 credit basis: worse than the 25% adverse threshold.
        QuoteGateState(entry_net=1.20, current_net=1.65, dte=30, deterministic_exit_reason=None),
    )

    assert gate.allowed is True
    assert "adverse" in gate.reason


def test_gate_refuses_non_finite_quote_values() -> None:
    for entry_net, current_net in (
        (math.nan, 1.65),
        (math.inf, 1.65),
        (-math.inf, 1.65),
        (1.20, math.nan),
        (1.20, math.inf),
        (1.20, -math.inf),
    ):
        gate = evaluate_exit_request_gate(
            _base(),
            QuoteGateState(
                entry_net=entry_net,
                current_net=current_net,
                dte=30,
                deterministic_exit_reason=None,
            ),
        )
        assert gate.allowed is False
        assert "finite" in gate.reason


def test_gate_refuses_blank_reason_and_case_duplicate_sources() -> None:
    quote = QuoteGateState(
        entry_net=1.20,
        current_net=1.65,
        dte=30,
        deterministic_exit_reason=None,
    )
    blank_reason = evaluate_exit_request_gate(_base(reason="   "), quote)
    duplicate_sources = evaluate_exit_request_gate(
        _base(sources=["Reuters", " reuters "]), quote
    )

    assert blank_reason.allowed is False
    assert "reason" in blank_reason.reason
    assert duplicate_sources.allowed is False
    assert "distinct" in duplicate_sources.reason


def test_gate_allows_deterministic_exit_even_without_two_news_sources() -> None:
    gate = evaluate_exit_request_gate(
        _base(sources=["bot deterministic exit"], catalyst_type="risk_management"),
        QuoteGateState(
            entry_net=1.20, current_net=0.80, dte=21, deterministic_exit_reason="time exit (21 DTE)"
        ),
    )

    assert gate.allowed is True
    assert "time exit" in gate.reason


def test_gate_enforces_daily_caps() -> None:
    gate = evaluate_exit_request_gate(
        _base(today_portfolio_requests=2),
        QuoteGateState(entry_net=1.20, current_net=2.00, dte=30, deterministic_exit_reason=None),
    )

    assert gate.allowed is False
    assert "portfolio/day" in gate.reason


def test_hermes_loss_cap_fails_closed_without_current_session_baseline() -> None:
    decision = evaluate_hermes_loss_cap(
        cumulative_realized_pnl=-50.0,
        day_start_net_liq=None,
        max_daily_loss_pct=0.02,
    )

    assert decision.allowed is False
    assert decision.evaluable is False
    assert "baseline" in decision.reason


def test_hermes_loss_cap_fails_closed_for_non_finite_inputs() -> None:
    cases = [
        {"cumulative_realized_pnl": math.nan, "day_start_net_liq": 10_000.0,
         "max_daily_loss_pct": 0.02},
        {"cumulative_realized_pnl": -50.0, "day_start_net_liq": math.nan,
         "max_daily_loss_pct": 0.02},
        {"cumulative_realized_pnl": -50.0, "day_start_net_liq": math.inf,
         "max_daily_loss_pct": 0.02},
        {"cumulative_realized_pnl": -50.0, "day_start_net_liq": 10_000.0,
         "max_daily_loss_pct": math.inf},
    ]

    for case in cases:
        decision = evaluate_hermes_loss_cap(**case)
        assert decision.allowed is False
        assert decision.evaluable is False
        assert decision.cap_dollars is None
        assert "non-finite" in decision.reason


def test_hermes_loss_cap_blocks_at_daily_loss_limit() -> None:
    decision = evaluate_hermes_loss_cap(
        cumulative_realized_pnl=-200.0,
        day_start_net_liq=10_000.0,
        max_daily_loss_pct=0.02,
    )

    assert decision.allowed is False
    assert decision.evaluable is True
    assert decision.cap_dollars == 200.0
    assert "breached" in decision.reason


def test_hermes_loss_cap_allows_when_cumulative_pnl_is_above_limit() -> None:
    decision = evaluate_hermes_loss_cap(
        cumulative_realized_pnl=-199.99,
        day_start_net_liq=10_000.0,
        max_daily_loss_pct=0.02,
    )

    assert decision.allowed is True
    assert decision.evaluable is True
    assert decision.cap_dollars == 200.0
