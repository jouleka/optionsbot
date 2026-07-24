"""Persistent correctness circuit breaker for Hermes-vetted entries.

This module deliberately depends only on SQLAlchemy and the storage schema so
the read-only restricted MCP can report the same metric without importing any
broker or execution code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, case, func, insert, select, update

from optionsbot.storage.schema import (
    entry_reviews,
    fills,
    hermes_overlay_state,
    orders,
    pick_outcomes,
    snapshots,
    strategy_scores,
)

MIN_JUDGEABLE = 20
ACCURACY_THRESHOLD = 0.5
STATE_ID = 1


@dataclass(frozen=True, slots=True)
class HermesOverlayState:
    enabled: bool
    reason: str | None
    ts: datetime | None
    judgeable: int
    accuracy: float | None


def _review_outcome_rows(engine: Engine) -> list[Any]:
    first_orders = (
        select(
            orders.c.strategy_score_id,
            func.min(orders.c.staged_ts).label("first_order_ts"),
        )
        .where(orders.c.intent == "open")
        .where(orders.c.strategy_score_id.is_not(None))
        .group_by(orders.c.strategy_score_id)
        .subquery()
    )
    with engine.connect() as conn:
        return list(
            conn.execute(
                select(
                    entry_reviews.c.id.label("review_id"),
                    entry_reviews.c.strategy_score_id,
                    entry_reviews.c.verdict,
                    entry_reviews.c.status,
                    entry_reviews.c.reason,
                    entry_reviews.c.reviewed_at,
                    strategy_scores.c.strategy,
                    strategy_scores.c.score,
                    strategy_scores.c.legs_json,
                    strategy_scores.c.suggestion_json,
                    snapshots.c.symbol,
                    first_orders.c.first_order_ts,
                    pick_outcomes.c.expiry,
                    pick_outcomes.c.terminal_spot,
                    pick_outcomes.c.realized_pnl,
                    pick_outcomes.c.win,
                    pick_outcomes.c.evaluated_at,
                )
                .select_from(
                    entry_reviews.join(
                        strategy_scores,
                        entry_reviews.c.strategy_score_id == strategy_scores.c.id,
                    )
                    .join(snapshots, strategy_scores.c.snapshot_id == snapshots.c.id)
                    .outerjoin(
                        first_orders,
                        entry_reviews.c.strategy_score_id == first_orders.c.strategy_score_id,
                    )
                    .outerjoin(
                        pick_outcomes,
                        entry_reviews.c.strategy_score_id == pick_outcomes.c.strategy_score_id,
                    )
                )
                .order_by(entry_reviews.c.reviewed_at.desc())
            ).fetchall()
        )


def _review_context(row: Any) -> tuple[bool, bool, str]:
    suggestion = row.suggestion_json if isinstance(row.suggestion_json, dict) else {}
    evidence = suggestion.get("review_evidence")
    evidence_ready = isinstance(evidence, dict) and evidence.get("ready") is True
    post_trade = (
        row.first_order_ts is not None
        and row.reviewed_at is not None
        and row.first_order_ts <= row.reviewed_at
    )
    if post_trade:
        return evidence_ready, True, "post_trade_observation"
    if not evidence_ready:
        return False, False, "operational_guard"
    return True, False, "pre_trade_forecast"


def _actual_round_trip_results(engine: Engine) -> list[dict[str, Any]]:
    """Completed fill cashflows by trade, commissions included."""
    fill_cash = (
        select(
            fills.c.order_id,
            func.sum(
                case(
                    (fills.c.side == "SELL", fills.c.price * fills.c.qty * 100.0),
                    else_=-(fills.c.price * fills.c.qty * 100.0),
                )
                - func.coalesce(fills.c.commission, 0.0)
            ).label("cash"),
        )
        .group_by(fills.c.order_id)
        .having(func.count() == func.count(fills.c.commission))
        .subquery()
    )
    entries = orders.alias("learning_entries")
    closes = orders.alias("learning_closes")
    entry_cash = fill_cash.alias("learning_entry_cash")
    close_cash = fill_cash.alias("learning_close_cash")
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                entries.c.strategy_score_id,
                entries.c.symbol,
                entries.c.strategy,
                closes.c.id.label("close_order_id"),
                closes.c.last_error.label("exit_reason"),
                (entry_cash.c.cash + close_cash.c.cash).label("actual_pnl"),
            )
            .select_from(
                entries.join(
                    closes,
                    closes.c.closes_order_id == entries.c.id,
                )
                .join(entry_cash, entry_cash.c.order_id == entries.c.id)
                .join(close_cash, close_cash.c.order_id == closes.c.id)
            )
            .where(entries.c.intent == "open")
            .where(entries.c.status == "filled")
            .where(closes.c.intent == "close")
            .where(closes.c.status == "filled")
            .where(entries.c.strategy_score_id.is_not(None))
        ).fetchall()
    return [
        {
            "pick_id": int(row.strategy_score_id),
            "symbol": str(row.symbol),
            "strategy": str(row.strategy),
            "pnl": float(row.actual_pnl),
            "close_order_id": int(row.close_order_id),
            "exit_reason": row.exit_reason,
        }
        for row in rows
        if row.actual_pnl is not None
    ]


def _terminal_call_summary(engine: Engine) -> dict[str, Any]:
    """Aggregate every settled candidate by entry-thesis outcome.

    This is intentionally broader than actual fills: the scanner produces many
    forward-tested 0DTE hypotheses, so Hermes can learn which strategy/symbol
    families have worked without needing the broker to risk money on each one.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                pick_outcomes.c.symbol,
                pick_outcomes.c.strategy,
                pick_outcomes.c.win,
                pick_outcomes.c.realized_pnl,
            )
        ).fetchall()

    def grouped(field: str) -> dict[str, dict[str, int | float]]:
        groups: dict[str, dict[str, int | float]] = {}
        for row in rows:
            key = str(getattr(row, field))
            group = groups.setdefault(
                key,
                {"calls": 0, "wins": 0, "losses": 0, "net_pnl": 0.0},
            )
            group["calls"] += 1
            group["wins" if bool(row.win) else "losses"] += 1
            group["net_pnl"] += float(row.realized_pnl)
        for group in groups.values():
            calls = int(group["calls"])
            group["win_rate"] = float(group["wins"]) / calls if calls else 0.0
            group["net_pnl"] = round(float(group["net_pnl"]), 2)
            group["avg_pnl"] = round(float(group["net_pnl"]) / calls, 2) if calls else 0.0
        return groups

    wins = sum(bool(row.win) for row in rows)
    net_pnl = sum(float(row.realized_pnl) for row in rows)
    return {
        "calls": len(rows),
        "wins": wins,
        "losses": len(rows) - wins,
        "win_rate": wins / len(rows) if rows else None,
        "net_pnl": round(net_pnl, 2),
        "avg_pnl": round(net_pnl / len(rows), 2) if rows else None,
        "by_strategy": grouped("strategy"),
        "by_symbol": grouped("symbol"),
    }


def learning_feedback(engine: Engine, *, recent_limit: int = 20) -> dict[str, Any]:
    """Actual and counterfactual lessons fed back to every Hermes review pass.

    Completed fill cashflows are the truth about money made or lost. Expiry-close
    counterfactuals are separately the truth about whether the original call
    ultimately worked. Keeping both prevents a profitable early exit from
    teaching "good call" when the thesis later failed, or a premature stop from
    teaching "bad call" when the structure ultimately won.
    """
    rows = _review_outcome_rows(engine)
    actual_results = _actual_round_trip_results(engine)
    actual_by_pick = {result["pick_id"]: result for result in actual_results}
    evaluated = [
        row
        for row in rows
        if row.win is not None or int(row.strategy_score_id) in actual_by_pick
    ]
    lessons: list[dict[str, Any]] = []
    mistakes = guarded_winners = 0
    forecast_judgeable = forecast_useful = 0
    for row in evaluated:
        evidence_ready, post_trade, context = _review_context(row)
        approved = row.verdict == "vetted_paper_candidate"
        # A bot fill is ground truth regardless of Hermes's verdict. This is
        # especially important for reviews that arrived after the order was
        # already filled: Hermes must learn the realized result without that
        # hindsight observation being scored as a pre-trade forecast.
        actual_result = actual_by_pick.get(int(row.strategy_score_id))
        filled_pnl = actual_result["pnl"] if actual_result is not None else None
        call_pnl = float(row.realized_pnl) if row.realized_pnl is not None else None
        call_won = bool(row.win) if row.win is not None else None
        execution_won = filled_pnl > 0.0 if filled_pnl is not None else None
        diagnosis: str | None = None
        if call_won is not None and execution_won is not None:
            diagnosis = {
                (True, True): "good_call_good_execution",
                (True, False): "good_call_bad_execution",
                (False, True): "bad_call_good_execution",
                (False, False): "bad_call_bad_execution",
            }[(call_won, execution_won)]
        observed_pnl = (
            call_pnl
            if context == "pre_trade_forecast" and call_pnl is not None
            else filled_pnl if filled_pnl is not None else call_pnl
        )
        assert observed_pnl is not None
        won = observed_pnl > 0.0
        outcome_basis = (
            "actual_filled_round_trip" if filled_pnl is not None else "expiry_close_counterfactual"
        )
        useful: bool | None = None
        lesson = "not_forecast_judgeable"
        if context == "pre_trade_forecast":
            forecast_judgeable += 1
            useful = won == approved
            forecast_useful += int(useful)
            if not useful:
                mistakes += 1
                lesson = "missed_winner" if won else "approved_loser"
            else:
                lesson = "correct_approval" if approved else "correct_rejection"
        elif context == "operational_guard" and won:
            guarded_winners += 1
            lesson = "theoretical_winner_but_execution_guard_was_not_overruled"
        elif context == "post_trade_observation" and diagnosis is not None:
            lesson = diagnosis
        elif context == "post_trade_observation" and filled_pnl is not None:
            lesson = (
                "actual_trade_winner_post_trade_observation"
                if execution_won
                else "actual_trade_loser_post_trade_observation"
            )
        lessons.append(
            {
                "review_id": int(row.review_id),
                "pick_id": int(row.strategy_score_id),
                "symbol": row.symbol,
                "strategy": row.strategy,
                "score": float(row.score),
                "verdict": row.verdict,
                "review_context": context,
                "evidence_ready": evidence_ready,
                "post_trade": post_trade,
                "expiry": row.expiry,
                "theoretical_win": bool(row.win) if row.win is not None else None,
                "theoretical_pnl": (
                    float(row.realized_pnl) if row.realized_pnl is not None else None
                ),
                "call_pnl": call_pnl,
                "call_won": call_won,
                "actual_trade_pnl": filled_pnl,
                "execution_won": execution_won,
                "close_order_id": (
                    actual_result["close_order_id"] if actual_result is not None else None
                ),
                "exit_reason": (
                    actual_result["exit_reason"] if actual_result is not None else None
                ),
                "diagnosis": diagnosis,
                "outcome_basis": outcome_basis,
                "observed_win": won,
                "observed_pnl": observed_pnl,
                "forecast_useful": useful,
                "lesson": lesson,
                "review_reason": row.reason,
            }
        )
    by_strategy: dict[str, dict[str, int | float]] = {}
    by_symbol: dict[str, dict[str, int | float]] = {}
    for result in actual_results:
        for key, groups in (
            (result["strategy"], by_strategy),
            (result["symbol"], by_symbol),
        ):
            group = groups.setdefault(
                key,
                {"trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0},
            )
            group["trades"] += 1
            group["wins" if result["pnl"] > 0 else "losses"] += 1
            group["net_pnl"] += result["pnl"]
    for groups in (by_strategy, by_symbol):
        for group in groups.values():
            trades = int(group["trades"])
            group["win_rate"] = float(group["wins"]) / trades if trades else 0.0
            group["net_pnl"] = round(float(group["net_pnl"]), 2)

    actual_wins = sum(result["pnl"] > 0 for result in actual_results)
    return {
        "meaning": (
            "Actual fill P&L (including commissions) measures execution; expiry-close "
            "counterfactual P&L measures call quality. Learn the correct side of a "
            "disagreement, but never waive stale-quote, liquidity, or risk guards."
        ),
        "review_calls": len(rows),
        "outcomes_available": len(evaluated),
        "forecast_judgeable": forecast_judgeable,
        "forecast_useful": forecast_useful,
        "forecast_accuracy": (forecast_useful / forecast_judgeable if forecast_judgeable else None),
        "forecast_mistakes": mistakes,
        "guarded_theoretical_winners": guarded_winners,
        "actual_trade_summary": {
            "trades": len(actual_results),
            "wins": actual_wins,
            "losses": len(actual_results) - actual_wins,
            "win_rate": (actual_wins / len(actual_results) if actual_results else None),
            "net_pnl": round(sum(result["pnl"] for result in actual_results), 2),
            "by_strategy": by_strategy,
            "by_symbol": by_symbol,
        },
        "terminal_call_summary": _terminal_call_summary(engine),
        "recent_lessons": lessons[: max(1, min(recent_limit, 100))],
    }


def correctness_report(engine: Engine) -> dict[str, Any]:
    """Return the single canonical correctness calculation."""
    rows = _review_outcome_rows(engine)

    verdicts = ("no_trade", "vetted_paper_candidate", "watch_only")
    by_verdict: dict[str, dict[str, int | float | None]] = {}
    judgeable = 0
    useful = 0
    for verdict in verdicts:
        matching = [row for row in rows if row.verdict == verdict]
        judged = [
            row
            for row in matching
            if row.win is not None and _review_context(row)[2] == "pre_trade_forecast"
        ]
        verdict_useful = sum(
            1 for row in judged if bool(row.win) == (verdict == "vetted_paper_candidate")
        )
        judgeable += len(judged)
        useful += verdict_useful
        by_verdict[verdict] = {
            "calls": len(matching),
            "judgeable": len(judged),
            "useful": verdict_useful,
            "accuracy": verdict_useful / len(judged) if judged else None,
        }

    accuracy = useful / judgeable if judgeable else None
    if accuracy is None:
        recommendation = "CHANGE"
    elif accuracy < ACCURACY_THRESHOLD:
        recommendation = "DISABLE"
    else:
        recommendation = "KEEP"
    return {
        "calls": len(rows),
        "judgeable": judgeable,
        "useful": useful,
        "accuracy": accuracy,
        "threshold": ACCURACY_THRESHOLD,
        "recommendation": recommendation,
        "small_sample": judgeable < MIN_JUDGEABLE,
        "unmatched_calls": len(rows) - judgeable,
        "by_verdict": by_verdict,
    }


def load_overlay_state(engine: Engine) -> HermesOverlayState:
    with engine.connect() as conn:
        row = conn.execute(
            select(hermes_overlay_state).where(hermes_overlay_state.c.id == STATE_ID)
        ).first()
    if row is None:
        return HermesOverlayState(
            False,
            "Hermes overlay state is missing; migrate or explicitly reset before entry",
            None,
            0,
            None,
        )
    return HermesOverlayState(
        enabled=bool(row.enabled),
        reason=row.reason,
        ts=row.ts,
        judgeable=int(row.judgeable),
        accuracy=float(row.accuracy) if row.accuracy is not None else None,
    )


def _persist(engine: Engine, state: HermesOverlayState) -> None:
    values = {
        "enabled": int(state.enabled),
        "reason": state.reason,
        "ts": state.ts,
        "judgeable": state.judgeable,
        "accuracy": state.accuracy,
    }
    with engine.begin() as conn:
        result = conn.execute(
            update(hermes_overlay_state)
            .where(hermes_overlay_state.c.id == STATE_ID)
            .values(**values)
        )
        if result.rowcount == 0:
            conn.execute(insert(hermes_overlay_state).values(id=STATE_ID, **values))


def evaluate_overlay(
    engine: Engine, *, now: datetime | None = None
) -> tuple[HermesOverlayState, bool]:
    """Evaluate new evidence, returning state and whether this call tripped it.

    Once disabled, the breaker never auto-enables even if later aggregate
    accuracy recovers.  An explicit reset acknowledges all current evidence;
    the next newly judgeable outcome causes a fresh evaluation.
    """
    current = load_overlay_state(engine)
    report = correctness_report(engine)
    judgeable = int(report["judgeable"])
    accuracy = report["accuracy"]
    if judgeable == current.judgeable:
        return current, False

    checked_at = now or datetime.now(UTC)
    enabled = current.enabled
    reason = current.reason
    tripped = False
    if (
        enabled
        and judgeable >= MIN_JUDGEABLE
        and accuracy is not None
        and float(accuracy) < ACCURACY_THRESHOLD
    ):
        enabled = False
        tripped = True
        reason = (
            f"Hermes overlay accuracy {float(accuracy):.1%} is below "
            f"{ACCURACY_THRESHOLD:.0%} after {judgeable} judgeable outcomes"
        )
    state = HermesOverlayState(
        enabled=enabled,
        reason=reason,
        ts=checked_at,
        judgeable=judgeable,
        accuracy=float(accuracy) if accuracy is not None else None,
    )
    _persist(engine, state)
    return state, tripped


def reset_overlay(engine: Engine, *, now: datetime | None = None) -> HermesOverlayState:
    """Explicitly re-enable the overlay and acknowledge current evidence."""
    report = correctness_report(engine)
    accuracy = report["accuracy"]
    state = HermesOverlayState(
        enabled=True,
        reason=None,
        ts=now or datetime.now(UTC),
        judgeable=int(report["judgeable"]),
        accuracy=float(accuracy) if accuracy is not None else None,
    )
    _persist(engine, state)
    return state


def hold_pending_reviews(engine: Engine, state: HermesOverlayState) -> int:
    """Terminally hold every unconsumed vetted review while disabled."""
    if state.enabled:
        return 0
    now = datetime.now(UTC)
    message = state.reason or "Hermes overlay correctness breaker is disabled"
    with engine.begin() as conn:
        result = conn.execute(
            update(entry_reviews)
            .where(entry_reviews.c.verdict == "vetted_paper_candidate")
            .where(entry_reviews.c.status.in_(("requested", "processing")))
            .values(
                status="held",
                decision_reason=f"overlay breaker: {message}",
                claimed_at=None,
                processed_at=now,
            )
        )
    return int(result.rowcount)


def breaker_report(engine: Engine) -> dict[str, Any]:
    state = load_overlay_state(engine)
    return {
        "enabled": state.enabled,
        "reason": state.reason,
        "changed_at": state.ts.isoformat() if state.ts is not None else None,
        "last_evaluated_judgeable": state.judgeable,
        "last_evaluated_accuracy": state.accuracy,
        "minimum_judgeable": MIN_JUDGEABLE,
        "accuracy_threshold": ACCURACY_THRESHOLD,
        "reset_required": not state.enabled,
    }
