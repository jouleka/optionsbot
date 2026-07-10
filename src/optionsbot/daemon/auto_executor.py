"""Full-auto entry hook (IBK-130): alerted candidates → execute_pick.

Runs the SAME pipeline as Telegram /execute — every gate (freshness, caps,
liquidity, margin, dedup, plus the auto-only earnings and buying-power
gates) applies per pick. Confirm mode never enters here.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import or_, select, update

from optionsbot.daemon.context import DaemonContext
from optionsbot.ibkr.market_data import MarketDataClient
from optionsbot.ibkr.positions import PositionsClient
from optionsbot.scoring import ScoredStrategy
from optionsbot.storage.schema import entry_reviews, snapshots, strategy_scores

log = logging.getLogger(__name__)


def _score_id_for(context: DaemonContext, snapshot_id: int, strategy: str) -> int | None:
    with context.engine.connect() as conn:
        row = conn.execute(
            select(strategy_scores.c.id)
            .where(strategy_scores.c.snapshot_id == snapshot_id)
            .where(strategy_scores.c.strategy == strategy)
            .order_by(strategy_scores.c.id.desc())
            .limit(1)
        ).first()
    return int(row.id) if row is not None else None


def _requested_review_id_for(context: DaemonContext, score_id: int) -> int | None:
    with context.engine.connect() as conn:
        row = conn.execute(
            select(entry_reviews.c.id)
            .where(entry_reviews.c.strategy_score_id == score_id)
            .where(entry_reviews.c.verdict == "vetted_paper_candidate")
            .where(entry_reviews.c.status == "requested")
            .limit(1)
        ).first()
    return int(row.id) if row is not None else None


def _claim_review(context: DaemonContext, review_id: int) -> bool:
    with context.engine.begin() as conn:
        result = conn.execute(
            update(entry_reviews)
            .where(entry_reviews.c.id == review_id)
            .where(entry_reviews.c.status == "requested")
            .values(
                status="processing",
                decision_reason=None,
                claimed_at=datetime.now(UTC),
            )
        )
    return result.rowcount == 1


def _finish_review(
    context: DaemonContext,
    review_id: int,
    *,
    ok: bool,
    message: str,
    order_id: int | None,
    failure_status: str = "held",
) -> None:
    values: dict[str, object] = {
        "decision_reason": message,
        "claimed_at": None,
    }
    if ok:
        values.update(
            status="submitted",
            processed_at=datetime.now(UTC),
            order_id=order_id,
        )
    else:
        values.update(
            status=failure_status,
            order_id=None,
            processed_at=datetime.now(UTC),
        )
    with context.engine.begin() as conn:
        conn.execute(
            update(entry_reviews)
            .where(entry_reviews.c.id == review_id)
            .where(entry_reviews.c.status == "processing")
            .values(**values)
        )


async def _execute_reviewed_score(
    context: DaemonContext,
    *,
    score_id: int,
    symbol: str,
    strategy: str,
) -> bool:
    """Claim and execute one exact reviewed score through ``execute_pick``."""
    if context.order_client is None:
        log.warning("auto-execute hold score_id=%s: order client unavailable", score_id)
        return False
    review_id = _requested_review_id_for(context, score_id)
    if review_id is None:
        log.info(
            "auto-execute hold %s/%s score_id=%s: awaiting exact Hermes review",
            symbol,
            strategy,
            score_id,
        )
        return False
    if not _claim_review(context, review_id):
        log.info("auto-execute hold review_id=%s: already claimed", review_id)
        return False

    try:
        # Imported lazily because the engine pulls daemon.market_hours; a
        # module-level import would close a cycle.
        from optionsbot.execution import engine as execution_engine

        walk_md = (
            MarketDataClient(context.exec_ibkr, context.resolver)
            if context.exec_ibkr is not None
            else None
        )
        deps = execution_engine.ExecutionDeps(
            engine=context.engine,
            settings=context.settings,
            order_client=context.order_client,
            md=MarketDataClient(context.ibkr, context.resolver),
            positions=PositionsClient(context.ibkr),
            ibkr_lock=context.ibkr_lock,
            walk_md=walk_md,
            walk_tasks=context.walk_tasks,
        )
        outcome = await execution_engine.execute_pick(deps, score_id)
        _finish_review(
            context,
            review_id,
            ok=outcome.ok,
            message=outcome.message,
            order_id=outcome.order_id,
        )
        log.info(
            "auto-execute %s/%s score_id=%s -> ok=%s | %s",
            symbol,
            strategy,
            score_id,
            outcome.ok,
            outcome.message.replace("\n", " "),
        )
        await _send(
            context,
            f"🤖 auto-execute {symbol} {strategy}:\n{outcome.message}",
        )
        return bool(outcome.ok)
    except Exception as exc:  # noqa: BLE001 -- one candidate must not starve the rest
        _finish_review(
            context,
            review_id,
            ok=False,
            message=f"unexpected auto-execution error: {type(exc).__name__}",
            order_id=None,
            failure_status="failed",
        )
        log.exception("auto-execute failed for %s/%s score_id=%s", symbol, strategy, score_id)
        return False


async def auto_execute_candidates(
    context: DaemonContext,
    candidates: list[tuple[str, ScoredStrategy, int]],
) -> int:
    """Execute only alerted candidates with an exact requested Hermes review."""
    if context.settings.execution.mode != "auto" or context.order_client is None:
        return 0
    submitted = 0
    log.info("auto-execute pass: %d candidate(s)", len(candidates))
    for symbol, scored, snapshot_id in candidates:
        score_id = _score_id_for(context, snapshot_id, scored.strategy_name)
        if score_id is None:
            log.warning(
                "auto-execute skip %s/%s: no score_id for snapshot %s",
                symbol,
                scored.strategy_name,
                snapshot_id,
            )
            continue
        if await _execute_reviewed_score(
            context,
            score_id=score_id,
            symbol=symbol,
            strategy=scored.strategy_name,
        ):
            submitted += 1
    return submitted


async def run_entry_reviews_tick(context: DaemonContext) -> int:
    """Consume delayed Hermes reviews through the normal auto-execution path."""
    if context.settings.execution.mode != "auto" or context.order_client is None:
        return 0
    now = datetime.now(UTC)
    cutoff = now - timedelta(minutes=context.settings.execution.max_pick_age_minutes)
    stale_score_ids = (
        select(strategy_scores.c.id)
        .join(snapshots, strategy_scores.c.snapshot_id == snapshots.c.id)
        .where(snapshots.c.ts < cutoff)
    )
    with context.engine.begin() as conn:
        conn.execute(
            update(entry_reviews)
            .where(entry_reviews.c.status == "processing")
            .where(
                or_(
                    entry_reviews.c.claimed_at.is_(None),
                    entry_reviews.c.claimed_at < now - timedelta(minutes=10),
                )
            )
            .values(
                status="requested",
                claimed_at=None,
                decision_reason="recovered abandoned processing lease",
            )
        )
        conn.execute(
            update(entry_reviews)
            .where(entry_reviews.c.verdict == "vetted_paper_candidate")
            .where(entry_reviews.c.status == "requested")
            .where(entry_reviews.c.strategy_score_id.in_(stale_score_ids))
            .values(
                status="expired",
                decision_reason="original candidate exceeded max_pick_age_minutes",
                processed_at=now,
            )
        )
    with context.engine.connect() as conn:
        rows = conn.execute(
            select(
                entry_reviews.c.id,
                strategy_scores.c.id.label("score_id"),
                strategy_scores.c.strategy,
                snapshots.c.symbol,
            )
            .join(
                strategy_scores,
                entry_reviews.c.strategy_score_id == strategy_scores.c.id,
            )
            .join(snapshots, strategy_scores.c.snapshot_id == snapshots.c.id)
            .where(entry_reviews.c.verdict == "vetted_paper_candidate")
            .where(entry_reviews.c.status == "requested")
            .order_by(entry_reviews.c.reviewed_at)
            .limit(20)
        ).fetchall()
    submitted = 0
    for row in rows:
        if await _execute_reviewed_score(
            context,
            score_id=int(row.score_id),
            symbol=str(row.symbol),
            strategy=str(row.strategy),
        ):
            submitted += 1
    return submitted


async def _send(context: DaemonContext, text: str) -> None:
    try:
        await context.telegram.send_message(text, parse_mode=None)
    except Exception:  # noqa: BLE001
        log.exception("auto-execute notification failed")
