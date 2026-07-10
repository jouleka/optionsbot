"""Tests for the net-liq drawdown circuit breaker (Phase 0, work-stream B)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine

from optionsbot.config import Settings
from optionsbot.execution.equity_guard import (
    EquityVerdict,
    capture_day_start_net_liq,
    evaluate_net_liq_drawdown,
    new_entry_allowed,
)
from optionsbot.execution.state import load_state


def _settings(*, daily: float = 0.02, block_frac: float = 0.75) -> Settings:
    s = Settings()
    s.execution.max_daily_loss_pct = daily
    s.execution.entry_block_loss_frac = block_frac
    return s


def test_capture_persists_and_is_idempotent(tmp_db: Engine) -> None:
    # First capture stores; a second capture for the same day must NOT overwrite
    # (so an intraday restart can't reset the high-water-mark and hide a loss).
    first = capture_day_start_net_liq(tmp_db, 100_000.0)
    assert first == 100_000.0
    again = capture_day_start_net_liq(tmp_db, 90_000.0)
    assert again == 100_000.0  # unchanged


def test_drawdown_below_cap_does_not_trip(tmp_db: Engine) -> None:
    capture_day_start_net_liq(tmp_db, 100_000.0)
    verdict = evaluate_net_liq_drawdown(
        tmp_db, _settings(), current_net_liq=99_000.0, now=datetime.now(UTC)
    )
    assert verdict.tripped is False
    assert load_state(tmp_db).killed is False


def test_drawdown_at_cap_trips_kill_with_unrealized_loss(tmp_db: Engine) -> None:
    # Nothing closed; a 2% mark-to-market decline alone must trip the kill.
    capture_day_start_net_liq(tmp_db, 100_000.0)
    verdict = evaluate_net_liq_drawdown(
        tmp_db, _settings(daily=0.02), current_net_liq=98_000.0, now=datetime.now(UTC)
    )
    assert verdict.tripped is True
    state = load_state(tmp_db)
    assert state.killed is True
    assert "net liq" in (state.reason or "").lower()


def test_drawdown_no_day_start_is_noop(tmp_db: Engine) -> None:
    # No captured baseline (never set this session) -> cannot evaluate, no trip.
    verdict = evaluate_net_liq_drawdown(
        tmp_db, _settings(), current_net_liq=50_000.0, now=datetime.now(UTC)
    )
    assert verdict.tripped is False
    assert verdict.evaluable is False
    assert load_state(tmp_db).killed is False


def test_drawdown_none_current_is_noop(tmp_db: Engine) -> None:
    capture_day_start_net_liq(tmp_db, 100_000.0)
    verdict = evaluate_net_liq_drawdown(
        tmp_db, _settings(), current_net_liq=None, now=datetime.now(UTC)
    )
    assert verdict.tripped is False
    assert verdict.evaluable is False


def test_new_entry_blocked_at_block_frac(tmp_db: Engine) -> None:
    # cap = 2%, block at 75% of cap = 1.5% drawdown. 1.6% down -> blocked.
    capture_day_start_net_liq(tmp_db, 100_000.0)
    blocked = new_entry_allowed(
        tmp_db, _settings(daily=0.02, block_frac=0.75), current_net_liq=98_400.0
    )
    assert blocked.allowed is False
    assert "drawdown" in blocked.reason.lower()


def test_new_entry_allowed_below_block_frac(tmp_db: Engine) -> None:
    capture_day_start_net_liq(tmp_db, 100_000.0)
    ok = new_entry_allowed(
        tmp_db, _settings(daily=0.02, block_frac=0.75), current_net_liq=99_000.0
    )
    assert ok.allowed is True


def test_new_entry_blocked_when_not_evaluable(tmp_db: Engine) -> None:
    # Missing baseline/current equity cannot prove the daily-loss guard, so new
    # risk must fail closed even though exits remain available.
    decision = new_entry_allowed(tmp_db, _settings(), current_net_liq=None)
    assert decision.allowed is False
    assert "not evaluable" in decision.reason


def test_evaluate_is_idempotent_when_already_killed(tmp_db: Engine) -> None:
    from optionsbot.execution.state import trip_kill

    capture_day_start_net_liq(tmp_db, 100_000.0)
    trip_kill(tmp_db, "prior")
    verdict = evaluate_net_liq_drawdown(
        tmp_db, _settings(), current_net_liq=98_000.0, now=datetime.now(UTC)
    )
    # Already killed -> reports not-newly-tripped, leaves the existing reason.
    assert verdict.tripped is False
    assert load_state(tmp_db).reason == "prior"


def test_equity_verdict_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    v = EquityVerdict(tripped=False, evaluable=True, drawdown_pct=0.0, reason="ok")
    with pytest.raises(FrozenInstanceError):
        v.tripped = True  # type: ignore[misc]
