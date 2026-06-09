"""Tests for the /record Telegram command (IBK-117)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from sqlalchemy import insert

from optionsbot.daemon.commands import dispatch
from optionsbot.storage.schema import pick_outcomes, snapshots, strategy_scores


def _ctx(daemon_engine) -> MagicMock:
    ctx = MagicMock()
    ctx.engine = daemon_engine
    return ctx


async def test_cmd_record_empty(daemon_engine) -> None:
    replies = await dispatch(_ctx(daemon_engine), "/record")
    assert "no evaluated outcomes yet" in replies[0].text


async def test_cmd_record_reports(daemon_engine) -> None:
    with daemon_engine.begin() as conn:
        snap_id = conn.execute(insert(snapshots).values(
            symbol="SPY", ts=datetime.now(UTC), spot=400.0)).inserted_primary_key[0]
        score_id = conn.execute(insert(strategy_scores).values(
            snapshot_id=snap_id, strategy="iron_condor", score=70.0)).inserted_primary_key[0]
        conn.execute(insert(pick_outcomes).values(
            strategy_score_id=score_id, symbol="SPY", strategy="iron_condor", expiry="20260101",
            entry_spot=400.0, predicted_prob_profit=0.6, score=70.0, credit_or_debit=80.0,
            max_profit=80.0, max_loss=420.0, risk_tier="balanced", terminal_spot=405.0,
            realized_pnl=80.0, win=1, evaluated_at=datetime.now(UTC),
        ))
    replies = await dispatch(_ctx(daemon_engine), "/record")
    assert "track record" in replies[0].text and "iron_condor" in replies[0].text
