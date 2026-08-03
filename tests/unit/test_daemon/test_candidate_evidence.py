"""Tests for the trusted daemon-to-Hermes evidence handoff."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import insert, select

from optionsbot.daemon.candidate_evidence import (
    capture_candidate_evidence,
    with_reconciled_economics,
)
from optionsbot.daemon.context import DaemonContext
from optionsbot.execution.equity_guard import capture_day_start_net_liq
from optionsbot.ibkr.types import AccountSummary, OptionQuote
from optionsbot.review_evidence import review_evidence_ready
from optionsbot.storage.schema import snapshots, strategy_scores
from optionsbot.strategies import StrategySuggestion

LEGS = [
    {
        "symbol": "SPY",
        "side": "sell",
        "sec_type": "OPT",
        "expiry": "20260828",
        "strike": 580.0,
        "right": "P",
        "quantity": 1,
    },
    {
        "symbol": "SPY",
        "side": "buy",
        "sec_type": "OPT",
        "expiry": "20260828",
        "strike": 575.0,
        "right": "P",
        "quantity": 1,
    },
]
NOW = datetime(2026, 7, 16, 15, 0, tzinfo=UTC)


def test_with_reconciled_economics_replaces_frozen_suggestion() -> None:
    original = StrategySuggestion(
        strategy_name="bull_put_spread",
        legs=(),
        credit_or_debit=100.0,
        max_loss=400.0,
        max_profit=100.0,
        prob_profit=0.65,
        suggested_quantity=2,
        defined_risk=True,
        rationale="scan economics",
        reward_risk=0.25,
        expected_value=10.0,
    )

    reconciled = with_reconciled_economics(
        original,
        {
            "economics": {
                "credit_or_debit": 120.0,
                "max_loss": 380.0,
                "max_profit": 120.0,
                "reward_risk": 120 / 380,
                "expected_value": 30.0,
            }
        },
    )

    assert reconciled is not original
    assert reconciled.credit_or_debit == 120.0
    assert reconciled.max_loss == 380.0
    assert reconciled.max_profit == 120.0
    assert reconciled.reward_risk == 120 / 380
    assert reconciled.expected_value == 30.0
    assert original.credit_or_debit == 100.0
    assert original.expected_value == 10.0


def _quote(
    strike: float,
    *,
    bid: float,
    ask: float,
    ts: datetime = NOW,
) -> OptionQuote:
    return OptionQuote(
        symbol="SPY",
        expiry="20260828",
        strike=strike,
        right="P",
        bid=bid,
        ask=ask,
        last=(bid + ask) / 2,
        mid=(bid + ask) / 2,
        iv=0.25,
        delta=-0.25,
        gamma=0.02,
        theta=-0.03,
        vega=0.10,
        open_interest=500,
        volume=50,
        ts=ts,
        delayed=False,
    )


async def test_capture_candidate_evidence_persists_ready_packet(
    daemon_context: DaemonContext,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    daemon_context.settings.execution.enabled = True
    daemon_context.settings.execution.mode = "auto"
    capture_day_start_net_liq(daemon_context.engine, 100_000.0, session="2026-07-16")
    with daemon_context.engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY",
                    ts=NOW,
                    spot=600.0,
                    raw_json={
                        "delayed": False,
                        "warming_up": False,
                        "next_earnings_date": None,
                        "earnings_source": "unknown",
                        "beta_to_benchmark": 1.0,
                        "beta_benchmark": "SPY",
                    },
                )
            ).inserted_primary_key[0]
        )
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="bull_put_spread",
                    score=80.0,
                    legs_json=LEGS,
                    suggestion_json={
                        "defined_risk": True,
                        "credit_or_debit": 100.0,
                        "max_loss": 400.0,
                        "max_profit": 100.0,
                        "prob_profit": 0.65,
                        "expected_value": 10.0,
                        "suggested_quantity": 2,
                    },
                )
            ).inserted_primary_key[0]
        )

    md = MagicMock()
    md.get_option_review_snapshot = AsyncMock(
        side_effect=[
            _quote(580.0, bid=1.55, ask=1.65),
            _quote(575.0, bid=0.35, ask=0.45),
        ]
    )
    positions = MagicMock()
    positions.get_account_summary = AsyncMock(
        return_value=AccountSummary(
            net_liquidation=Decimal("100000"),
            buying_power=Decimal("100000"),
            available_funds=Decimal("100000"),
            currency="USD",
        )
    )
    positions.get_portfolio = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "optionsbot.daemon.candidate_evidence.MarketDataClient",
        MagicMock(return_value=md),
    )
    monkeypatch.setattr(
        "optionsbot.daemon.candidate_evidence.PositionsClient",
        MagicMock(return_value=positions),
    )

    evidence = await capture_candidate_evidence(
        daemon_context,
        score_id=score_id,
        symbol="SPY",
        legs=LEGS,
        now=NOW,
    )

    assert evidence["ready"] is True, evidence["readiness_issues"]
    assert evidence["readiness_issues"] == []
    assert len(evidence["option_quotes"]) == 2
    assert evidence["combo"]["mid"] == 1.2
    assert evidence["economics"]["credit_or_debit"] == pytest.approx(120.0)
    assert evidence["economics"]["max_loss"] == pytest.approx(380.0)
    assert evidence["economics"]["max_profit"] == pytest.approx(120.0)
    assert evidence["economics"]["expected_value"] == pytest.approx(30.0)
    assert evidence["risk"]["single_trade_risk_allowed"] is True
    assert evidence["risk"]["portfolio_heat_allowed"] is True
    assert evidence["schema_version"] == 2
    assert evidence["combo"]["spread_fraction_of_net_premium"] == pytest.approx(
        0.2 / 1.2
    )
    assert evidence["combo"]["spread_allowed"] is True
    assert evidence["candidate_greeks"]["complete"] is True
    assert evidence["candidate_greeks"]["net_delta_share_equivalent"] == 0.0
    assert evidence["exposure"]["complete"] is True
    assert evidence["exposure"]["beta_delta_hard_cap_configured"] is False
    assert evidence["exposure"][
        "incremental_beta_weighted_delta_pct_of_net_liq"
    ] == pytest.approx(0.0)
    assert evidence["risk"]["candidate_affordable"] is True
    assert evidence["risk"]["bp_deployment_allowed"] is True
    assert evidence["risk"]["opening_range_daily_entry_allowed"] is True
    assert evidence["risk"]["opening_range_session_entries"] == 0
    assert evidence["candidate_scope"] == {
        "review_authorization_units": 1,
        "strategy_units_reviewed": 1,
        "economics_scope": "one_strategy_unit",
        "risk_scope": "one_strategy_unit",
        "greeks_scope": "one_strategy_unit",
        "suggested_quantity": 2,
        "suggested_quantity_role": "non_authoritative_scan_hint",
        "execution_quantity_recomputed_by_daemon": True,
        "review_quantity_policy": (
            "review authorizes at most the proven one-unit candidate; the daemon "
            "independently sizes and reruns all aggregate risk gates"
        ),
    }
    assert evidence["market_timing"]["entry_window_open"] is True
    assert evidence["expiration_assignment"]["handling"]
    assert review_evidence_ready(
        evidence,
        score_id=score_id,
        now=NOW,
        max_age_minutes=20,
    )
    with daemon_context.engine.connect() as conn:
        suggestion = conn.execute(
            select(strategy_scores.c.suggestion_json).where(strategy_scores.c.id == score_id)
        ).scalar_one()
    assert suggestion["review_evidence"]["source"] == "trusted_daemon"
    assert suggestion["credit_or_debit"] == pytest.approx(120.0)
    assert suggestion["max_loss"] == pytest.approx(380.0)
    assert suggestion["max_profit"] == pytest.approx(120.0)
    assert suggestion["expected_value"] == pytest.approx(30.0)
