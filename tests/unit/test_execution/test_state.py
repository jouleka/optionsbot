"""Tests for the persisted execution kill switch (IBK-123)."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine

from optionsbot.execution.state import (
    clear_kill,
    is_session_loss_kill,
    load_state,
    trip_kill,
)
from optionsbot.storage.db import create_engine_for_path
from tests.conftest import apply_migrations  # noqa: TID252 (cross-package import OK in tests)


def test_fresh_db_is_not_killed(tmp_db: Engine) -> None:
    state = load_state(tmp_db)
    assert state.killed is False
    assert state.reason is None


def test_trip_kill_round_trips(tmp_db: Engine) -> None:
    trip_kill(tmp_db, "max daily loss breached")
    state = load_state(tmp_db)
    assert state.killed is True
    assert state.reason == "max daily loss breached"
    assert state.ts is not None
    # SQLite drops tzinfo on round-trip; load_state must re-attach UTC (same
    # defense as alert_dedup) so later phases can do aware-datetime arithmetic.
    assert state.ts.tzinfo is not None


def test_kill_survives_engine_restart(tmp_path: Path) -> None:
    # The whole point of persisting: a daemon restart must NOT silently re-arm.
    db_path = tmp_path / "restart.db"
    apply_migrations(db_path)
    first = create_engine_for_path(db_path)
    trip_kill(first, "crash drill")
    first.dispose()

    second = create_engine_for_path(db_path)
    state = load_state(second)
    assert state.killed is True
    assert state.reason == "crash drill"


def test_clear_kill_re_arms(tmp_db: Engine) -> None:
    trip_kill(tmp_db, "oops")
    clear_kill(tmp_db)
    state = load_state(tmp_db)
    assert state.killed is False
    assert state.reason is None


def test_trip_twice_keeps_latest_reason(tmp_db: Engine) -> None:
    trip_kill(tmp_db, "first")
    trip_kill(tmp_db, "second")
    assert load_state(tmp_db).reason == "second"


def test_only_session_loss_reasons_are_session_scoped() -> None:
    assert is_session_loss_kill("net liq drawdown 2.02% >= 2% cap")
    assert is_session_loss_kill("daily realized loss $250 ≥ 2% of net liq")
    assert is_session_loss_kill("3 consecutive losing trades this session")
    assert is_session_loss_kill(
        "daily cumulative Hermes realized-loss cap breached (-$200 <= -$175)"
    )
    assert not is_session_loss_kill("manual /kill via Telegram")
    assert not is_session_loss_kill("reconcile exact position mismatch")
    assert not is_session_loss_kill("broker side effects are uncertain")
