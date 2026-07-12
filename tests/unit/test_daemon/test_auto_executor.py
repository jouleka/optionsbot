"""Tests for the full-auto entry hook + loss kill-triggers (IBK-130)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import insert, select, text, update
from sqlalchemy.exc import IntegrityError

from optionsbot.daemon.auto_executor import auto_execute_candidates
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.market_hours import nyse_session_date
from optionsbot.daemon.order_watcher import run_orders_tick
from optionsbot.execution.engine import ExecuteOutcome
from optionsbot.execution.orders import record_fill, set_fill_commission
from optionsbot.execution.state import load_state
from optionsbot.storage.schema import (
    alerts,
    entry_intent_consumptions,
    entry_reviews,
    orders,
    snapshots,
    strategy_scores,
)

NOW = datetime.now(UTC)


def _pick(
    context: DaemonContext,
    symbol: str = "SPY",
    *,
    snapshot_ts: datetime = NOW,
) -> tuple[int, int]:
    with context.engine.begin() as conn:
        snap = int(conn.execute(insert(snapshots).values(
            symbol=symbol, ts=snapshot_ts, spot=600.0,
            raw_json={"delayed": False, "warming_up": False},
        )).inserted_primary_key[0])
        score = int(conn.execute(insert(strategy_scores).values(
            snapshot_id=snap, strategy="bull_put_spread", score=80.0,
            rationale="t",
            legs_json=[
                {
                    "symbol": symbol,
                    "side": "sell",
                    "sec_type": "OPT",
                    "expiry": "20260717",
                    "strike": 590.0,
                    "right": "P",
                    "quantity": 1,
                },
                {
                    "symbol": symbol,
                    "side": "buy",
                    "sec_type": "OPT",
                    "expiry": "20260717",
                    "strike": 585.0,
                    "right": "P",
                    "quantity": 1,
                },
            ],
            suggestion_json={
                "defined_risk": True,
                "credit_or_debit": 100.0,
                "max_loss": 400.0,
                "max_profit": 100.0,
                "prob_profit": 0.70,
                "expected_value": 15.0,
                "suggested_quantity": 1,
            },
        )).inserted_primary_key[0])
    return snap, score


def _alert(
    context: DaemonContext,
    score_id: int,
    *,
    status: str = "sent",
    sent_ts: datetime | None = NOW,
    telegram_msg_id: int | None = 12345,
) -> int:
    with context.engine.begin() as conn:
        row = conn.execute(
            select(strategy_scores.c.strategy, snapshots.c.symbol)
            .join(snapshots, strategy_scores.c.snapshot_id == snapshots.c.id)
            .where(strategy_scores.c.id == score_id)
        ).one()
        pk = conn.execute(
            insert(alerts).values(
                strategy_score_id=score_id,
                ts=NOW,
                symbol=row.symbol,
                strategy=row.strategy,
                score=80.0,
                status=status,
                sent_ts=sent_ts if status == "sent" else None,
                telegram_msg_id=telegram_msg_id if status == "sent" else None,
            )
        ).inserted_primary_key
    return int(pk[0])


def _review(
    context: DaemonContext,
    score_id: int,
    *,
    alert_id: int | None = None,
    create_alert: bool = True,
    confidence: float = 0.90,
    sources: list[str] | None = None,
    checks: dict[str, bool] | None = None,
    reviewed_at: datetime = NOW,
) -> int:
    if alert_id is None and create_alert:
        alert_id = _alert(context, score_id)
    if checks is None:
        checks = {
            "bot_health": True,
            "candidate": True,
            "microstructure": True,
            "greeks": True,
            "regime_history": True,
            "catalysts": True,
            "account_risk": True,
        }
    with context.engine.begin() as conn:
        pk = conn.execute(
            insert(entry_reviews).values(
                strategy_score_id=score_id,
                alert_id=alert_id,
                reviewed_at=reviewed_at,
                verdict="vetted_paper_candidate",
                confidence=confidence,
                sources_json=sources or ["source A", "source B"],
                reason="test review",
                checks_json=checks,
                status="requested",
            )
        ).inserted_primary_key
    return int(pk[0])


async def test_auto_does_not_execute_unreviewed_candidate(
    daemon_context: DaemonContext,
) -> None:
    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    snap, _ = _pick(daemon_context)
    scored = MagicMock()
    scored.strategy_name = "bull_put_spread"

    with patch("optionsbot.execution.engine.execute_pick", new=AsyncMock()) as run:
        n = await auto_execute_candidates(daemon_context, [("SPY", scored, snap)])

    assert n == 0
    run.assert_not_awaited()


async def test_daemon_rejects_malformed_persisted_review_even_if_ingress_was_bypassed(
    daemon_context: DaemonContext,
) -> None:
    """The order-capable consumer is the final trust boundary."""
    from optionsbot.daemon.auto_executor import run_entry_reviews_tick

    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    _, score = _pick(daemon_context)
    with daemon_context.engine.begin() as conn:
        conn.exec_driver_sql("DROP TRIGGER trg_entry_reviews_validate_insert")
    _review(
        daemon_context,
        score,
        create_alert=False,
        confidence=0.0,
        sources=["one source"],
        checks={},
    )

    with patch("optionsbot.execution.engine.execute_pick", new=AsyncMock()) as run:
        n = await run_entry_reviews_tick(daemon_context)

    assert n == 0
    run.assert_not_awaited()
    with daemon_context.engine.connect() as conn:
        review = conn.execute(select(entry_reviews)).one()
    assert review.status == "held"
    assert "invalid authorization" in review.decision_reason


async def test_daemon_rejects_slightly_future_review_timestamp(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.daemon.auto_executor import run_entry_reviews_tick

    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    _, score_id = _pick(daemon_context)
    _review(
        daemon_context,
        score_id,
        reviewed_at=datetime.now(UTC) + timedelta(seconds=30),
    )

    with patch("optionsbot.execution.engine.execute_pick", new=AsyncMock()) as run:
        n = await run_entry_reviews_tick(daemon_context)

    assert n == 0
    run.assert_not_awaited()
    with daemon_context.engine.connect() as conn:
        review = conn.execute(select(entry_reviews)).one()
    assert review.status == "held"
    assert "invalid authorization" in review.decision_reason


async def test_daemon_rejects_review_whose_alert_links_another_score(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.daemon.auto_executor import run_entry_reviews_tick

    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    _, reviewed_score = _pick(daemon_context, "SPY")
    _, other_score = _pick(daemon_context, "QQQ")
    wrong_alert = _alert(daemon_context, other_score)
    with daemon_context.engine.begin() as conn:
        conn.exec_driver_sql("DROP TRIGGER trg_entry_reviews_validate_insert")
    _review(
        daemon_context,
        reviewed_score,
        alert_id=wrong_alert,
        create_alert=False,
    )

    with patch("optionsbot.execution.engine.execute_pick", new=AsyncMock()) as run:
        n = await run_entry_reviews_tick(daemon_context)

    assert n == 0
    run.assert_not_awaited()
    with daemon_context.engine.connect() as conn:
        review = conn.execute(select(entry_reviews)).one()
    assert review.status == "held"


async def test_daemon_rejects_wrong_score_alert_with_identical_metadata(
    daemon_context: DaemonContext,
) -> None:
    """Alert identity, not matching display fields, authorizes the exact score."""
    from optionsbot.daemon.auto_executor import run_entry_reviews_tick

    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    _, reviewed_score = _pick(daemon_context, "SPY")
    _, lookalike_score = _pick(daemon_context, "SPY")
    wrong_alert = _alert(daemon_context, lookalike_score)
    with daemon_context.engine.begin() as conn:
        conn.exec_driver_sql("DROP TRIGGER trg_entry_reviews_validate_insert")
    _review(
        daemon_context,
        reviewed_score,
        alert_id=wrong_alert,
        create_alert=False,
    )

    with patch("optionsbot.execution.engine.execute_pick", new=AsyncMock()) as run:
        submitted = await run_entry_reviews_tick(daemon_context)

    assert submitted == 0
    run.assert_not_awaited()
    with daemon_context.engine.connect() as conn:
        review = conn.execute(select(entry_reviews)).one()
    assert review.status == "held"


async def test_daemon_requires_review_after_proven_alert_delivery(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.daemon.auto_executor import run_entry_reviews_tick

    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    _, score = _pick(daemon_context)
    alert_id = _alert(
        daemon_context,
        score,
        sent_ts=NOW + timedelta(minutes=5),
        telegram_msg_id=None,
    )
    with daemon_context.engine.begin() as conn:
        conn.execute(text("DROP TRIGGER trg_entry_reviews_validate_insert"))
    _review(daemon_context, score, alert_id=alert_id, create_alert=False)

    with patch("optionsbot.execution.engine.execute_pick", new=AsyncMock()) as run:
        submitted = await run_entry_reviews_tick(daemon_context)

    assert submitted == 0
    run.assert_not_awaited()


async def test_daemon_derives_defined_risk_from_legs_not_suggestion_flag(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.daemon.auto_executor import run_entry_reviews_tick

    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    _, score = _pick(daemon_context)
    with daemon_context.engine.begin() as conn:
        conn.execute(
            update(strategy_scores)
            .where(strategy_scores.c.id == score)
            .values(
                legs_json=[
                    {
                        "symbol": "SPY",
                        "side": "sell",
                        "sec_type": "OPT",
                        "expiry": "20260717",
                        "strike": 590.0,
                        "right": "P",
                        "quantity": 1,
                    }
                ]
            )
        )
    _review(daemon_context, score)

    with patch("optionsbot.execution.engine.execute_pick", new=AsyncMock()) as run:
        submitted = await run_entry_reviews_tick(daemon_context)

    assert submitted == 0
    run.assert_not_awaited()


async def test_auto_executes_alerted_candidates(daemon_context: DaemonContext) -> None:
    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    snap, score = _pick(daemon_context)
    _review(daemon_context, score)
    scored = MagicMock()
    scored.strategy_name = "bull_put_spread"
    with patch(
        "optionsbot.execution.engine.execute_pick",
        new=AsyncMock(
            return_value=ExecuteOutcome(ok=True, message="✅ submitted", order_id=None)
        ),
    ) as run:
        n = await auto_execute_candidates(daemon_context, [("SPY", scored, snap)])
    assert n == 1
    assert run.await_args.args[1] == score
    with daemon_context.engine.connect() as conn:
        review = conn.execute(select(entry_reviews)).one()
    assert review.status == "submitted"
    assert review.order_id is None
    assert review.processed_at is not None
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("🤖" in m and "submitted" in m for m in sent)


async def test_review_tick_executes_review_submitted_after_scan(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.daemon.auto_executor import run_entry_reviews_tick

    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    _, score = _pick(daemon_context)
    _review(daemon_context, score)

    with patch(
        "optionsbot.execution.engine.execute_pick",
        new=AsyncMock(return_value=ExecuteOutcome(ok=True, message="submitted", order_id=None)),
    ) as run:
        n = await run_entry_reviews_tick(daemon_context)

    assert n == 1
    assert run.await_args.args[1] == score
    with daemon_context.engine.connect() as conn:
        review = conn.execute(select(entry_reviews)).one()
    assert review.status == "submitted"
    assert review.order_id is None


async def test_strategy_identity_is_unique_within_snapshot(
    daemon_context: DaemonContext,
) -> None:
    snap, _ = _pick(daemon_context)

    with pytest.raises(IntegrityError):
        with daemon_context.engine.begin() as conn:
            conn.execute(
                insert(strategy_scores).values(
                    snapshot_id=snap,
                    strategy="bull_put_spread",
                    score=81.0,
                    rationale="ambiguous duplicate strategy row",
                    legs_json=[],
                    suggestion_json={},
                )
            )


async def test_review_tick_executes_exact_reviewed_score_id(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.daemon.auto_executor import run_entry_reviews_tick

    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    _, reviewed_score = _pick(daemon_context)
    _review(daemon_context, reviewed_score)

    with patch(
        "optionsbot.execution.engine.execute_pick",
        new=AsyncMock(
            return_value=ExecuteOutcome(ok=True, message="submitted", order_id=None)
        ),
    ) as run:
        n = await run_entry_reviews_tick(daemon_context)

    assert n == 1
    assert run.await_args.args[1] == reviewed_score


async def test_review_rejection_becomes_held_and_is_not_retried(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.daemon.auto_executor import run_entry_reviews_tick

    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    _, score = _pick(daemon_context)
    _review(daemon_context, score)
    rejected = ExecuteOutcome(ok=False, message="NO TRADE: spread too wide", order_id=None)

    with patch(
        "optionsbot.execution.engine.execute_pick",
        new=AsyncMock(return_value=rejected),
    ) as run:
        first = await run_entry_reviews_tick(daemon_context)
        second = await run_entry_reviews_tick(daemon_context)

    assert first == 0
    assert second == 0
    assert run.await_count == 1
    with daemon_context.engine.connect() as conn:
        review = conn.execute(select(entry_reviews)).one()
    assert review.status == "held"
    assert review.processed_at is not None
    assert "spread too wide" in review.decision_reason


async def test_review_tick_expires_stale_pick_without_execution(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.daemon.auto_executor import run_entry_reviews_tick

    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    _snap, score = _pick(
        daemon_context,
        snapshot_ts=NOW - timedelta(minutes=31),
    )
    _review(daemon_context, score)

    with patch("optionsbot.execution.engine.execute_pick", new=AsyncMock()) as run:
        n = await run_entry_reviews_tick(daemon_context)

    assert n == 0
    run.assert_not_awaited()
    with daemon_context.engine.connect() as conn:
        review = conn.execute(select(entry_reviews)).one()
    assert review.status == "expired"
    assert review.processed_at is not None


async def test_review_tick_does_not_recover_claim_after_any_prior_order_intent(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.daemon.auto_executor import run_entry_reviews_tick

    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    _, score = _pick(daemon_context)
    review_id = _review(daemon_context, score)
    with daemon_context.engine.begin() as conn:
        conn.execute(
            update(entry_reviews)
            .where(entry_reviews.c.id == review_id)
            .values(status="processing", claimed_at=NOW - timedelta(minutes=11))
        )
        conn.execute(
            insert(orders).values(
                id=9,
                intent="open",
                strategy_score_id=score,
                symbol="SPY",
                strategy="bull_put_spread",
                legs_json=[],
                quantity=1,
                status="rejected",
                staged_ts=NOW,
                terminal_ts=NOW,
                reprice_count=0,
            )
        )
        conn.execute(
            insert(entry_intent_consumptions).values(
                strategy_score_id=score,
                first_order_id=9,
                consumed_at=NOW,
            )
        )
        # Even if a legacy order's nullable attribution is damaged, the
        # immutable consumption receipt must keep the candidate spent.
        conn.execute(
            update(orders).where(orders.c.id == 9).values(strategy_score_id=None)
        )

    with patch("optionsbot.execution.engine.execute_pick", new=AsyncMock()) as run:
        n = await run_entry_reviews_tick(daemon_context)

    assert n == 0
    run.assert_not_awaited()
    with daemon_context.engine.connect() as conn:
        review = conn.execute(select(entry_reviews)).one()
    assert review.status == "held"
    assert review.claimed_at is None
    assert "prior order intent" in review.decision_reason


async def test_review_tick_recovers_unconsumed_abandoned_claim(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.daemon.auto_executor import run_entry_reviews_tick

    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    _, score = _pick(daemon_context)
    review_id = _review(daemon_context, score)
    with daemon_context.engine.begin() as conn:
        conn.execute(
            update(entry_reviews)
            .where(entry_reviews.c.id == review_id)
            .values(status="processing", claimed_at=NOW - timedelta(minutes=11))
        )

    rejected = ExecuteOutcome(ok=False, message="NO TRADE: final gates refused", order_id=None)
    with patch(
        "optionsbot.execution.engine.execute_pick",
        new=AsyncMock(return_value=rejected),
    ) as run:
        n = await run_entry_reviews_tick(daemon_context)

    assert n == 0
    run.assert_awaited_once()
    with daemon_context.engine.connect() as conn:
        review = conn.execute(select(entry_reviews)).one()
    assert review.status == "held"


async def test_review_completion_race_trips_kill_switch(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.daemon.auto_executor import run_entry_reviews_tick

    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    _, score = _pick(daemon_context)
    review_id = _review(daemon_context, score)

    async def race_review_state(*_args: object, **_kwargs: object) -> ExecuteOutcome:
        with daemon_context.engine.begin() as conn:
            conn.execute(
                update(entry_reviews)
                .where(entry_reviews.c.id == review_id)
                .values(status="held", decision_reason="simulated concurrent mutation")
            )
        return ExecuteOutcome(ok=True, message="submitted #9", order_id=None)

    with patch(
        "optionsbot.execution.engine.execute_pick",
        new=AsyncMock(side_effect=race_review_state),
    ):
        n = await run_entry_reviews_tick(daemon_context)

    assert n == 0
    state = load_state(daemon_context.engine)
    assert state.killed is True
    assert "review" in (state.reason or "").lower()


async def test_review_completion_exception_after_side_effect_trips_kill_switch(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.daemon.auto_executor import run_entry_reviews_tick

    daemon_context.settings.execution.mode = "auto"
    daemon_context.order_client = MagicMock()
    _, score = _pick(daemon_context)
    _review(daemon_context, score)

    with patch(
        "optionsbot.execution.engine.execute_pick",
        new=AsyncMock(
            return_value=ExecuteOutcome(ok=True, message="submitted #9", order_id=None)
        ),
    ), patch(
        "optionsbot.daemon.auto_executor._finish_review",
        side_effect=RuntimeError("sqlite unavailable"),
    ):
        submitted = await run_entry_reviews_tick(daemon_context)

    assert submitted == 0
    state = load_state(daemon_context.engine)
    assert state.killed is True
    assert "review" in (state.reason or "").lower()


async def test_auto_noop_in_confirm_mode(daemon_context: DaemonContext) -> None:
    daemon_context.order_client = MagicMock()
    snap, _ = _pick(daemon_context)
    scored = MagicMock()
    scored.strategy_name = "bull_put_spread"
    with patch(
        "optionsbot.execution.engine.execute_pick", new=AsyncMock()
    ) as run:
        n = await auto_execute_candidates(daemon_context, [("SPY", scored, snap)])
    assert n == 0
    run.assert_not_awaited()


# --- loss kill-triggers (order watcher) -----------------------------------------------


def _closed_pair(context: DaemonContext, *, pnl_credit: float, closed_ts: datetime) -> int:
    engine = context.engine
    entry_leg = {
        "symbol": "SPY",
        "side": "sell",
        "sec_type": "OPT",
        "expiry": "20260731",
        "strike": 580.0,
        "right": "P",
        "quantity": 1,
        "con_id": 580001,
        "multiplier": 100,
        "currency": "USD",
    }
    with engine.begin() as conn:
        epk = int(conn.execute(insert(orders).values(
            intent="open", symbol="SPY", strategy="bull_put_spread",
            legs_json=[entry_leg],
            quantity=1, status="filled", staged_ts=closed_ts - timedelta(days=3),
            terminal_ts=closed_ts - timedelta(days=3), reprice_count=0,
        )).inserted_primary_key[0])
        cpk = int(conn.execute(insert(orders).values(
            intent="close", closes_order_id=epk, symbol="SPY",
            strategy="bull_put_spread", legs_json=[{**entry_leg, "side": "buy"}],
            quantity=1, status="filled",
            staged_ts=closed_ts, terminal_ts=closed_ts, reprice_count=0,
        )).inserted_primary_key[0])
        for oid in (epk, cpk):
            conn.execute(update(orders).where(orders.c.id == oid)
                         .values(order_ref=f"obot-{oid}"))
    # entry collects pnl_credit, close costs 0 -> pair pnl = pnl_credit*100 - commissions
    record_fill(engine, epk, exec_id=f"k{epk}", side="SELL",
                price=max(pnl_credit, 0.01) if pnl_credit > 0 else 0.10,
                qty=1, ts=closed_ts - timedelta(days=3), leg_con_id=580001)
    close_price = 0.01 if pnl_credit > 0 else 0.10 - pnl_credit
    record_fill(engine, cpk, exec_id=f"k{cpk}", side="BUY",
                price=close_price, qty=1, ts=closed_ts, leg_con_id=580001)
    set_fill_commission(engine, f"k{epk}", 0.65)
    set_fill_commission(engine, f"k{cpk}", 0.65)
    return cpk


async def test_consecutive_losses_trip_kill(daemon_context: DaemonContext) -> None:
    daemon_context.settings.execution.max_consecutive_losses = 2
    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()
    daemon_context.orders_notified_through = NOW - timedelta(hours=1)
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(minutes=5))
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(minutes=1))
    with patch("optionsbot.daemon.order_watcher._net_liq", new=AsyncMock(return_value=100_000.0)):
        await run_orders_tick(daemon_context)
    assert load_state(daemon_context.engine).killed is True
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("consecutive" in m.lower() for m in sent)


async def test_daily_loss_trips_kill(daemon_context: DaemonContext) -> None:
    daemon_context.settings.execution.max_consecutive_losses = 99
    daemon_context.settings.execution.max_daily_loss_pct = 0.02
    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()
    daemon_context.orders_notified_through = NOW - timedelta(hours=1)
    # One big loss today: -25/unit x 100 = -2,501.3 incl. commissions on 100k = -2.5%.
    _closed_pair(daemon_context, pnl_credit=-25.0, closed_ts=NOW - timedelta(minutes=1))
    with patch("optionsbot.daemon.order_watcher._net_liq", new=AsyncMock(return_value=100_000.0)):
        await run_orders_tick(daemon_context)
    assert load_state(daemon_context.engine).killed is True
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("daily loss" in m.lower() for m in sent)


async def test_winning_closes_do_not_kill(daemon_context: DaemonContext) -> None:
    daemon_context.settings.execution.max_consecutive_losses = 2
    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()
    daemon_context.orders_notified_through = NOW - timedelta(hours=1)
    _closed_pair(daemon_context, pnl_credit=1.0, closed_ts=NOW - timedelta(minutes=1))
    with patch("optionsbot.daemon.order_watcher._net_liq", new=AsyncMock(return_value=100_000.0)):
        await run_orders_tick(daemon_context)
    assert load_state(daemon_context.engine).killed is False


async def test_unrealized_drawdown_trips_kill_with_nothing_closed(
    daemon_context: DaemonContext,
) -> None:
    # PHASE 0 B1: no closed pairs at all — purely a mark-to-market decline past
    # the cap must trip the kill on a normal tick.
    from optionsbot.daemon.market_hours import nyse_session_date
    from optionsbot.execution.equity_guard import capture_day_start_net_liq

    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()
    # Pre-capture the day-start baseline keyed to TODAY's NYSE session so the
    # breaker sees the 100k baseline when the tick runs with mocked 97k net-liq.
    today_session = nyse_session_date(NOW).isoformat()
    capture_day_start_net_liq(daemon_context.engine, 100_000.0, session=today_session)
    daemon_context.day_start_net_liq = 100_000.0
    # 97k = 3% down, cap is 2% -> trip. Nothing closed (close_filled stays False).
    with patch(
        "optionsbot.daemon.order_watcher._net_liq",
        new=AsyncMock(return_value=97_000.0),
    ):
        await run_orders_tick(daemon_context)
    state = load_state(daemon_context.engine)
    assert state.killed is True
    assert "net liq" in (state.reason or "").lower()
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("net-liq" in m.lower() or "net liq" in m.lower() for m in sent)


async def test_unrealized_drawdown_under_cap_does_not_trip(
    daemon_context: DaemonContext,
) -> None:
    from optionsbot.execution.equity_guard import capture_day_start_net_liq

    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()
    capture_day_start_net_liq(
        daemon_context.engine, 100_000.0, session=nyse_session_date(NOW).isoformat()
    )
    with patch(
        "optionsbot.daemon.order_watcher._net_liq",
        new=AsyncMock(return_value=99_500.0),  # 0.5% down
    ):
        await run_orders_tick(daemon_context)
    assert load_state(daemon_context.engine).killed is False


async def test_daily_loss_window_uses_et_session_not_utc(
    daemon_context: DaemonContext,
) -> None:
    # PHASE 0 B2: a loss closed at 11:00 ET (15:00 UTC, 2026-06-15) must count
    # toward the SAME ET session as a 'now' of 20:30 ET (00:30 UTC, 2026-06-16).
    # A UTC-keyed day_start (00:00 UTC the 16th) would EXCLUDE the loss and the
    # kill would not trip; the ET anchor INCLUDES it and it DOES.
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    daemon_context.settings.execution.max_consecutive_losses = 99
    daemon_context.settings.execution.max_daily_loss_pct = 0.02
    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()

    now_utc = _dt(2026, 6, 15, 20, 30, tzinfo=et).astimezone(UTC)
    loss_close_utc = _dt(2026, 6, 15, 11, 0, tzinfo=et).astimezone(UTC)
    # Set watermark before the loss so the close shows as newly-terminal (triggers close_filled).
    daemon_context.orders_notified_through = loss_close_utc - timedelta(minutes=1)
    cpk = _closed_pair(daemon_context, pnl_credit=-25.0, closed_ts=loss_close_utc)
    with daemon_context.engine.begin() as conn:
        conn.execute(
            update(orders).where(orders.c.id == cpk).values(terminal_ts=loss_close_utc)
        )
    with patch(
        "optionsbot.daemon.order_watcher._net_liq",
        new=AsyncMock(return_value=100_000.0),
    ):
        await run_orders_tick(daemon_context, now=now_utc)
    assert load_state(daemon_context.engine).killed is True


async def test_consecutive_loss_ignores_prior_session_losses(
    daemon_context: DaemonContext,
) -> None:
    # PHASE 0 B3: two losses from YESTERDAY's session + one fresh loss today must
    # NOT trip a 3-consecutive-loss kill — only THIS session's losses count.
    daemon_context.settings.execution.max_consecutive_losses = 3
    daemon_context.settings.execution.max_daily_loss_pct = 0.99  # don't let daily-loss trip
    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()
    daemon_context.orders_notified_through = NOW - timedelta(days=2)
    # Two stale losses ~26-27h ago (prior ET session).
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(hours=27))
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(hours=26))
    # One fresh loss this session.
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(minutes=1))
    with patch(
        "optionsbot.daemon.order_watcher._net_liq",
        new=AsyncMock(return_value=100_000.0),
    ):
        await run_orders_tick(daemon_context)
    # Only ONE loss this session < limit of 3 -> NOT killed.
    assert load_state(daemon_context.engine).killed is False


async def test_consecutive_loss_trips_within_session(
    daemon_context: DaemonContext,
) -> None:
    # Three losses THIS session in a row -> trips, proving the session scoping
    # didn't simply disable the check.
    daemon_context.settings.execution.max_consecutive_losses = 3
    daemon_context.settings.execution.max_daily_loss_pct = 0.99
    daemon_context.order_client = MagicMock()
    daemon_context.order_client.cancel = AsyncMock()
    daemon_context.orders_notified_through = NOW - timedelta(hours=2)
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(minutes=20))
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(minutes=10))
    _closed_pair(daemon_context, pnl_credit=-1.0, closed_ts=NOW - timedelta(minutes=1))
    with patch(
        "optionsbot.daemon.order_watcher._net_liq",
        new=AsyncMock(return_value=100_000.0),
    ):
        await run_orders_tick(daemon_context)
    assert load_state(daemon_context.engine).killed is True
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("consecutive" in m.lower() for m in sent)
