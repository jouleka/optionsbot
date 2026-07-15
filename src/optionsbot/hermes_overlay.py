"""Persistent correctness circuit breaker for Hermes-vetted entries.

This module deliberately depends only on SQLAlchemy and the storage schema so
the read-only restricted MCP can report the same metric without importing any
broker or execution code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, insert, select, update

from optionsbot.storage.schema import (
    entry_reviews,
    hermes_overlay_state,
    pick_outcomes,
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


def correctness_report(engine: Engine) -> dict[str, Any]:
    """Return the single canonical correctness calculation."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                entry_reviews.c.verdict,
                entry_reviews.c.status,
                entry_reviews.c.reviewed_at,
                pick_outcomes.c.win,
                pick_outcomes.c.evaluated_at,
            ).select_from(
                entry_reviews.outerjoin(
                    pick_outcomes,
                    entry_reviews.c.strategy_score_id == pick_outcomes.c.strategy_score_id,
                )
            )
        ).fetchall()

    verdicts = ("no_trade", "vetted_paper_candidate", "watch_only")
    by_verdict: dict[str, dict[str, int | float | None]] = {}
    judgeable = 0
    useful = 0
    for verdict in verdicts:
        matching = [row for row in rows if row.verdict == verdict]
        judged = [row for row in matching if row.win is not None]
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
