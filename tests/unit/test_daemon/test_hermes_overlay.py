"""Correctness and persistence tests for the Hermes entry-overlay breaker."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import delete, insert

from optionsbot.daemon.context import DaemonContext
from optionsbot.hermes_overlay import evaluate_overlay, load_overlay_state, reset_overlay
from optionsbot.storage.schema import (
    alerts,
    entry_reviews,
    hermes_overlay_state,
    pick_outcomes,
    snapshots,
    strategy_scores,
)


def test_missing_overlay_state_fails_closed(daemon_context: DaemonContext) -> None:
    with daemon_context.engine.begin() as conn:
        conn.execute(delete(hermes_overlay_state))

    missing = load_overlay_state(daemon_context.engine)
    assert missing.enabled is False
    assert "missing" in str(missing.reason)

    reset = reset_overlay(daemon_context.engine)
    assert reset.enabled is True


def _add_judgeable(
    context: DaemonContext, *, count: int, wins: int, offset: int = 0
) -> None:
    now = datetime.now(UTC)
    with context.engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="SPY", ts=now, spot=600.0, raw_json={}
                )
            ).inserted_primary_key[0]
        )
        for index in range(count):
            strategy = f"breaker-test-{offset + index}"
            score_id = int(
                conn.execute(
                    insert(strategy_scores).values(
                        snapshot_id=snapshot_id,
                        strategy=strategy,
                        score=80.0,
                        legs_json=[],
                        suggestion_json={},
                    )
                ).inserted_primary_key[0]
            )
            alert_id = int(
                conn.execute(
                    insert(alerts).values(
                        strategy_score_id=score_id,
                        ts=now,
                        symbol="SPY",
                        strategy=strategy,
                        score=80.0,
                        status="sent",
                        sent_ts=now,
                        telegram_msg_id=offset + index + 1,
                    )
                ).inserted_primary_key[0]
            )
            conn.execute(
                insert(entry_reviews).values(
                    strategy_score_id=score_id,
                    alert_id=alert_id,
                    reviewed_at=now,
                    verdict="vetted_paper_candidate",
                    confidence=0.9,
                    sources_json=["source A", "source B"],
                    reason="breaker test review",
                    checks_json={
                        "bot_health": True,
                        "candidate": True,
                        "microstructure": True,
                        "greeks": True,
                        "regime_history": True,
                        "catalysts": True,
                        "account_risk": True,
                    },
                    status="submitted",
                )
            )
            win = int(index < wins)
            conn.execute(
                insert(pick_outcomes).values(
                    strategy_score_id=score_id,
                    symbol="SPY",
                    strategy=strategy,
                    expiry="2026-07-17",
                    entry_spot=600.0,
                    terminal_spot=601.0,
                    realized_pnl=100.0 if win else -100.0,
                    win=win,
                    evaluated_at=now,
                )
            )


def test_overlay_waits_for_minimum_sample(daemon_context: DaemonContext) -> None:
    _add_judgeable(daemon_context, count=19, wins=0)

    state, tripped = evaluate_overlay(daemon_context.engine)

    assert tripped is False
    assert state.enabled is True
    assert state.judgeable == 19
    assert state.accuracy == 0.0


def test_overlay_trips_persists_and_requires_explicit_reset(
    daemon_context: DaemonContext,
) -> None:
    _add_judgeable(daemon_context, count=20, wins=9)

    state, tripped = evaluate_overlay(daemon_context.engine)

    assert tripped is True
    assert state.enabled is False
    assert state.accuracy == 0.45
    assert "below 50%" in str(state.reason)
    assert load_overlay_state(daemon_context.engine).enabled is False

    reset = reset_overlay(daemon_context.engine)
    assert reset.enabled is True
    assert reset.judgeable == 20

    state, tripped = evaluate_overlay(daemon_context.engine)
    assert tripped is False
    assert state.enabled is True

    _add_judgeable(daemon_context, count=1, wins=0, offset=20)
    state, tripped = evaluate_overlay(daemon_context.engine)
    assert tripped is True
    assert state.enabled is False
    assert state.accuracy == 9 / 21

    _add_judgeable(daemon_context, count=12, wins=12, offset=21)
    state, tripped = evaluate_overlay(daemon_context.engine)
    assert tripped is False
    assert state.enabled is False
    assert state.accuracy == 21 / 33
