"""Correctness and persistence tests for the Hermes entry-overlay breaker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, insert

from optionsbot.daemon.context import DaemonContext
from optionsbot.hermes_overlay import (
    evaluate_overlay,
    learning_feedback,
    load_overlay_state,
    reset_overlay,
)
from optionsbot.storage.schema import (
    alerts,
    entry_reviews,
    fills,
    hermes_overlay_state,
    orders,
    pick_outcomes,
    position_settlements,
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


def _add_judgeable(context: DaemonContext, *, count: int, wins: int, offset: int = 0) -> None:
    now = datetime.now(UTC)
    with context.engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(symbol="SPY", ts=now, spot=600.0, raw_json={})
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
                        suggestion_json={"review_evidence": {"ready": True}},
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


def test_learning_feedback_records_actual_post_trade_result_without_outcome(
    daemon_context: DaemonContext,
) -> None:
    """A hindsight Hermes refusal still learns the bot's actual round trip."""
    now = datetime.now(UTC)
    entry_ts = now - timedelta(minutes=5)
    with daemon_context.engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(symbol="IWM", ts=entry_ts, spot=295.0, raw_json={})
            ).inserted_primary_key[0]
        )
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="bull_call_spread",
                    score=75.0,
                    legs_json=[],
                    suggestion_json={
                        "review_evidence": {"ready": True},
                        "credit_or_debit": -80.0,
                        "max_profit": 420.0,
                    },
                )
            ).inserted_primary_key[0]
        )
        alert_id = int(
            conn.execute(
                insert(alerts).values(
                    strategy_score_id=score_id,
                    ts=entry_ts,
                    symbol="IWM",
                    strategy="bull_call_spread",
                    score=75.0,
                    status="sent",
                    sent_ts=entry_ts,
                    telegram_msg_id=999,
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(entry_reviews).values(
                strategy_score_id=score_id,
                alert_id=alert_id,
                reviewed_at=now,
                verdict="no_trade",
                confidence=0.9,
                sources_json=["source A", "source B"],
                reason="review arrived after the fill",
                checks_json={},
                status="refused",
            )
        )
        entry_id = int(
            conn.execute(
                insert(orders).values(
                    strategy_score_id=score_id,
                    intent="open",
                    symbol="IWM",
                    strategy="bull_call_spread",
                    legs_json=[],
                    quantity=1,
                    status="filled",
                    staged_ts=entry_ts,
                    terminal_ts=entry_ts,
                )
            ).inserted_primary_key[0]
        )
        close_id = int(
            conn.execute(
                insert(orders).values(
                    intent="close",
                    symbol="IWM",
                    strategy="bull_call_spread",
                    legs_json=[],
                    quantity=1,
                    closes_order_id=entry_id,
                    status="filled",
                    staged_ts=now,
                    terminal_ts=now,
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(fills),
            [
                {
                    "order_id": entry_id,
                    "ib_exec_id": "hermes-entry",
                    "side": "BUY",
                    "price": 0.80,
                    "qty": 1,
                    "ts": entry_ts,
                    "commission": 1.0,
                },
                {
                    "order_id": close_id,
                    "ib_exec_id": "hermes-close",
                    "side": "SELL",
                    "price": 0.50,
                    "qty": 1,
                    "ts": now,
                    "commission": 1.0,
                },
            ],
        )

    feedback = learning_feedback(daemon_context.engine)

    assert feedback["outcomes_available"] == 1
    assert feedback["forecast_judgeable"] == 0
    assert feedback["actual_trade_summary"] == {
        "trades": 1,
        "wins": 0,
        "losses": 1,
        "win_rate": 0.0,
        "net_pnl": -32.0,
        "by_strategy": {
            "bull_call_spread": {
                "trades": 1,
                "wins": 0,
                "losses": 1,
                "net_pnl": -32.0,
                "win_rate": 0.0,
            }
        },
        "by_symbol": {
            "IWM": {
                "trades": 1,
                "wins": 0,
                "losses": 1,
                "net_pnl": -32.0,
                "win_rate": 0.0,
            }
        },
    }
    lesson = feedback["recent_lessons"][0]
    assert lesson["review_context"] == "post_trade_observation"
    assert lesson["outcome_basis"] == "actual_filled_round_trip"
    assert lesson["actual_trade_pnl"] == -32.0
    assert lesson["max_profit_at_entry"] == 420.0
    assert lesson["realized_profit_capture_pct"] == pytest.approx(-32.0 / 420.0)
    assert lesson["theoretical_pnl"] is None
    assert lesson["forecast_useful"] is None
    assert lesson["lesson"] == "actual_trade_loser_post_trade_observation"

    # Once the expiry-close counterfactual arrives, the learning loop must
    # separate entry-call quality from realized execution/exit quality.
    with daemon_context.engine.begin() as conn:
        conn.execute(
            insert(pick_outcomes).values(
                strategy_score_id=score_id,
                symbol="IWM",
                strategy="bull_call_spread",
                expiry="2026-07-24",
                entry_spot=295.0,
                terminal_spot=300.0,
                realized_pnl=150.0,
                win=1,
                evaluated_at=now,
            )
        )

    feedback = learning_feedback(daemon_context.engine)
    lesson = feedback["recent_lessons"][0]
    assert lesson["call_pnl"] == 150.0
    assert lesson["actual_trade_pnl"] == -32.0
    assert lesson["call_won"] is True
    assert lesson["execution_won"] is False
    assert lesson["diagnosis"] == "good_call_bad_execution"
    assert lesson["lesson"] == "good_call_bad_execution"
    assert feedback["terminal_call_summary"]["calls"] == 1
    assert feedback["terminal_call_summary"]["wins"] == 1
    assert feedback["terminal_call_summary"]["by_strategy"]["bull_call_spread"] == {
        "calls": 1,
        "wins": 1,
        "losses": 0,
        "net_pnl": 150.0,
        "win_rate": 1.0,
        "avg_pnl": 150.0,
    }


def test_learning_feedback_includes_intrinsic_expiration_settlement(
    daemon_context: DaemonContext,
) -> None:
    now = datetime.now(UTC)
    with daemon_context.engine.begin() as conn:
        snapshot_id = int(
            conn.execute(
                insert(snapshots).values(
                    symbol="NVDA",
                    ts=now,
                    spot=190.0,
                    raw_json={},
                )
            ).inserted_primary_key[0]
        )
        score_id = int(
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snapshot_id,
                    strategy="iron_butterfly",
                    score=80.0,
                    legs_json=[],
                    suggestion_json={},
                )
            ).inserted_primary_key[0]
        )
        entry_id = int(
            conn.execute(
                insert(orders).values(
                    strategy_score_id=score_id,
                    intent="open",
                    symbol="NVDA",
                    strategy="iron_butterfly",
                    legs_json=[],
                    quantity=1,
                    status="filled",
                    staged_ts=now,
                    terminal_ts=now,
                )
            ).inserted_primary_key[0]
        )
        conn.execute(
            insert(fills).values(
                order_id=entry_id,
                ib_exec_id="intrinsic-entry",
                side="SELL",
                price=2.70,
                qty=1,
                ts=now,
                commission=2.49,
            )
        )
        conn.execute(
            insert(position_settlements).values(
                entry_order_id=entry_id,
                kind="expired_intrinsic",
                expiry="20260729",
                terminal_spot=190.01,
                pnl=18.51,
                commissions=2.49,
                settled_at=now,
            )
        )

    feedback = learning_feedback(daemon_context.engine)

    summary = feedback["actual_trade_summary"]
    assert summary["trades"] == 1
    assert summary["wins"] == 1
    assert summary["net_pnl"] == 18.51
    assert summary["by_strategy"]["iron_butterfly"]["net_pnl"] == 18.51


def test_guarded_learning_is_payoff_aware_not_just_hit_rate(
    daemon_context: DaemonContext,
) -> None:
    now = datetime.now(UTC)
    with daemon_context.engine.begin() as conn:
        for index, pnl in enumerate((10.0, -100.0), start=1):
            snapshot_id = int(
                conn.execute(
                    insert(snapshots).values(
                        symbol="NVDA",
                        ts=now + timedelta(seconds=index),
                        spot=195.0,
                        raw_json={},
                    )
                ).inserted_primary_key[0]
            )
            score_id = int(
                conn.execute(
                    insert(strategy_scores).values(
                        snapshot_id=snapshot_id,
                        strategy="iron_condor",
                        score=70.0 + index,
                        legs_json=[],
                        suggestion_json={
                            "review_evidence": {
                                "ready": False,
                                "reason": "market data unavailable",
                            }
                        },
                    )
                ).inserted_primary_key[0]
            )
            alert_id = int(
                conn.execute(
                    insert(alerts).values(
                        strategy_score_id=score_id,
                        ts=now,
                        symbol="NVDA",
                        strategy="iron_condor",
                        score=70.0 + index,
                        status="sent",
                        sent_ts=now,
                        telegram_msg_id=1_100 + index,
                    )
                ).inserted_primary_key[0]
            )
            conn.execute(
                insert(entry_reviews).values(
                    strategy_score_id=score_id,
                    alert_id=alert_id,
                    reviewed_at=now,
                    verdict="no_trade",
                    confidence=0.9,
                    sources_json=[],
                    reason="evidence unavailable",
                    checks_json={"bot_health": False},
                    status="refused",
                )
            )
            conn.execute(
                insert(pick_outcomes).values(
                    strategy_score_id=score_id,
                    symbol="NVDA",
                    strategy="iron_condor",
                    expiry="2026-07-27",
                    entry_spot=195.0,
                    terminal_spot=196.0,
                    realized_pnl=pnl,
                    win=int(pnl > 0),
                    evaluated_at=now,
                )
            )

    summary = learning_feedback(daemon_context.engine)["guarded_call_summary"]

    assert summary["calls"] == 2
    assert summary["win_rate"] == 0.5
    assert summary["net_pnl"] == -90.0
    assert summary["avg_pnl"] == -45.0
    assert summary["avg_win"] == 10.0
    assert summary["avg_loss"] == -100.0
