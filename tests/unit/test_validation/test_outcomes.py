"""Tests for the forward outcome ledger (IBK-99 Phase B)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from alembic.config import Config
from sqlalchemy import insert, select

from alembic import command
from optionsbot.storage.db import create_engine_for_path
from optionsbot.storage.schema import pick_outcomes, snapshots, strategy_scores
from optionsbot.strategies.base import Leg
from optionsbot.validation.outcomes import (
    evaluate_pending,
    evaluate_pnl,
    outcomes_report,
)


def _migrated_engine(tmp_path):
    db_path = tmp_path / "o.db"
    root = Path(__file__).resolve().parents[3]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")
    return create_engine_for_path(db_path)


def test_migration_creates_pick_outcomes(tmp_path) -> None:
    engine = _migrated_engine(tmp_path)
    with engine.connect() as conn:
        rows = conn.execute(select(pick_outcomes)).fetchall()
    assert rows == []  # table exists and is empty


def test_evaluate_pnl_long_call_itm() -> None:
    legs = (Leg(symbol="X", side="buy", sec_type="OPT", expiry="20260101",
                strike=100.0, right="C", quantity=1),)
    # terminal 120 -> intrinsic 20*100=2000 + cod(-500 debit) = 1500 > 0 win
    pnl, win = evaluate_pnl(legs, credit_or_debit=-500.0, terminal_spot=120.0)
    assert pnl == 1500.0
    assert win is True
    # terminal 100 -> intrinsic 0 + cod(-500) = -500 loss
    pnl2, win2 = evaluate_pnl(legs, credit_or_debit=-500.0, terminal_spot=100.0)
    assert pnl2 == -500.0
    assert win2 is False


def _seed_expired_pick(engine, *, shadow_only: bool = False) -> None:
    with engine.begin() as conn:
        sid = conn.execute(insert(snapshots).values(
            symbol="SPY", ts=datetime(2026, 1, 1, tzinfo=UTC), spot=100.0
        )).inserted_primary_key[0]
        conn.execute(insert(strategy_scores).values(
            snapshot_id=sid, strategy="long_call", score=55.0,
            legs_json=[{"symbol": "SPY", "side": "buy", "sec_type": "OPT",
                        "expiry": "20260201", "strike": 100.0, "right": "C",
                        "quantity": 1}],
            suggestion_json={"prob_profit": 0.4, "credit_or_debit": -500.0,
                             "max_profit": None, "max_loss": 500.0,
                             "risk_tier": "aggressive",
                             "shadow_only": shadow_only,
                             "admission_enabled": not shadow_only}))


async def test_evaluate_pending_persists_and_dedups(tmp_path) -> None:
    engine = _migrated_engine(tmp_path)
    _seed_expired_pick(engine)

    async def fake_close(symbol: str, expiry: str) -> float:
        return 120.0  # ITM -> win, pnl = 1500

    today = date(2026, 3, 1)  # after the 20260201 expiry
    n = await evaluate_pending(engine, fake_close, today)
    assert n == 1
    with engine.connect() as conn:
        rows = conn.execute(select(pick_outcomes)).fetchall()
    assert len(rows) == 1
    assert rows[0].win == 1
    assert rows[0].realized_pnl == 1500.0
    assert rows[0].terminal_spot == 120.0
    # Second run is idempotent (UNIQUE(strategy_score_id) + the NULL-join filter).
    n2 = await evaluate_pending(engine, fake_close, today)
    assert n2 == 0


async def test_evaluate_pending_skips_unexpired(tmp_path) -> None:
    engine = _migrated_engine(tmp_path)
    _seed_expired_pick(engine)  # expiry 20260201

    async def fake_close(symbol: str, expiry: str) -> float:
        return 120.0

    # today BEFORE expiry -> nothing evaluated.
    assert await evaluate_pending(engine, fake_close, date(2026, 1, 15)) == 0


async def test_evaluate_pending_excludes_shadow_research_from_track_record(
    tmp_path,
) -> None:
    engine = _migrated_engine(tmp_path)
    _seed_expired_pick(engine, shadow_only=True)

    async def fake_close(symbol: str, expiry: str) -> float:
        raise AssertionError("shadow-only row must not request a terminal close")

    assert await evaluate_pending(engine, fake_close, date(2026, 3, 1)) == 0
    with engine.connect() as conn:
        assert conn.execute(select(pick_outcomes)).fetchall() == []


async def test_outcomes_report_aggregates(tmp_path) -> None:
    engine = _migrated_engine(tmp_path)
    _seed_expired_pick(engine)

    async def fake_close(symbol: str, expiry: str) -> float:
        return 120.0

    await evaluate_pending(engine, fake_close, date(2026, 3, 1))
    report = outcomes_report(engine)
    assert report.overall.count == 1
    assert report.overall.win_rate == 1.0
    assert report.overall.total_pnl == 1500.0
    assert "long_call" in report.by_strategy
    assert "aggressive" in report.by_risk_tier
