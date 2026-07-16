"""Tests for the trusted daemon-to-Hermes evidence handoff."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import insert, select

from optionsbot.daemon.candidate_evidence import capture_candidate_evidence
from optionsbot.daemon.context import DaemonContext
from optionsbot.execution.equity_guard import capture_day_start_net_liq
from optionsbot.ibkr.types import AccountSummary, OptionQuote
from optionsbot.review_evidence import review_evidence_ready
from optionsbot.storage.schema import snapshots, strategy_scores

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


def _quote(strike: float, *, bid: float, ask: float) -> OptionQuote:
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
        ts=datetime.now(UTC),
        delayed=False,
    )


async def test_capture_candidate_evidence_persists_ready_packet(
    daemon_context: DaemonContext,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    daemon_context.settings.execution.enabled = True
    daemon_context.settings.execution.mode = "auto"
    capture_day_start_net_liq(
        daemon_context.engine, 100_000.0, session="2026-07-16"
    )
    with daemon_context.engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY",
                    ts=datetime.now(UTC),
                    raw_json={"delayed": False, "warming_up": False},
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
                    suggestion_json={"defined_risk": True, "max_loss": 380.0},
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
    )

    assert evidence["ready"] is True, evidence["readiness_issues"]
    assert evidence["readiness_issues"] == []
    assert len(evidence["option_quotes"]) == 2
    assert evidence["combo"]["mid"] == 1.2
    assert review_evidence_ready(
        evidence,
        score_id=score_id,
        now=datetime.now(UTC),
        max_age_minutes=20,
    )
    with daemon_context.engine.connect() as conn:
        suggestion = conn.execute(
            select(strategy_scores.c.suggestion_json).where(strategy_scores.c.id == score_id)
        ).scalar_one()
    assert suggestion["review_evidence"]["source"] == "trusted_daemon"
