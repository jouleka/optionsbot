"""Alert pipeline: dedup, persist, dispatch, retry sweep (IBK-67)."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

from sqlalchemy import and_, insert, or_, select, update

from optionsbot.alerts import format_alert_markdown
from optionsbot.analysis.types import Direction, IVRegime, MarketView
from optionsbot.daemon.alert_dedup import should_alert
from optionsbot.daemon.context import DaemonContext
from optionsbot.scoring import ScoredStrategy
from optionsbot.scoring.types import FactorBreakdown
from optionsbot.storage.schema import alerts, snapshots, strategy_scores

log = logging.getLogger(__name__)

# Exponential backoff schedule for retries. Indexed by (new_count - 1):
# 1st failure -> BACKOFF_MINUTES[0] = 1m, 5th -> BACKOFF_MINUTES[4] = 240m.
BACKOFF_MINUTES: tuple[int, ...] = (1, 5, 15, 60, 240)
MAX_RETRIES: int = len(BACKOFF_MINUTES)


async def enqueue_alert(
    context: DaemonContext,
    symbol: str,
    scored: ScoredStrategy,
    snapshot_id: int,
) -> bool:
    """Dedup-check, insert pending row, dispatch.

    Returns True when an alerts row was inserted (and dispatch was attempted),
    False when the dedup gate suppressed it. The caller increments its
    enqueued counter only on True so dedup-skipped attempts don't pad
    scan_runs.alerts_fired with phantom rows.
    """
    if not should_alert(
        context.engine, context.settings, symbol, scored.strategy_name, scored.score
    ):
        return False
    now = datetime.now(UTC)
    strategy_score_id = _strategy_score_id_for(context, snapshot_id, scored.strategy_name)
    if strategy_score_id is None:
        raise RuntimeError(
            f"missing exact persisted score for snapshot={snapshot_id} "
            f"strategy={scored.strategy_name}"
        )
    with context.engine.begin() as conn:
        result = conn.execute(
            insert(alerts).values(
                strategy_score_id=strategy_score_id,
                ts=now,
                symbol=symbol,
                strategy=scored.strategy_name,
                score=scored.score,
                status="pending",
            )
        )
        alert_id = cast(int, result.inserted_primary_key[0])  # type: ignore[index]
    # Hermes is deliberately broker-isolated. Capture the exact live quote,
    # account, and risk packet before the alert becomes visible as sent.
    try:
        from optionsbot.daemon.candidate_evidence import (
            capture_candidate_evidence,
            with_reconciled_economics,
        )

        evidence = await capture_candidate_evidence(
            context,
            score_id=strategy_score_id,
            symbol=symbol,
            legs=[
                {
                    "symbol": leg.symbol,
                    "side": leg.side,
                    "sec_type": leg.sec_type,
                    "expiry": leg.expiry,
                    "strike": leg.strike,
                    "right": leg.right,
                    "quantity": leg.quantity,
                }
                for leg in scored.suggestion.legs
            ],
        )
        scored = replace(
            scored,
            suggestion=with_reconciled_economics(scored.suggestion, evidence),
        )
    except Exception:  # noqa: BLE001 - alert still delivers; Hermes fails closed
        log.exception("candidate evidence capture failed for score %s", strategy_score_id)
    await dispatch_alert(context, alert_id, snapshot_id, scored)
    return True


def _strategy_score_id_for(context: DaemonContext, snapshot_id: int, strategy: str) -> int | None:
    with context.engine.connect() as conn:
        row = conn.execute(
            select(strategy_scores.c.id)
            .where(strategy_scores.c.snapshot_id == snapshot_id)
            .where(strategy_scores.c.strategy == strategy)
        ).one_or_none()
    return cast(int, row.id) if row is not None else None


def execute_hint_for(context: DaemonContext, snapshot_id: int, strategy: str) -> str | None:
    """`/execute <strategy_scores.id>` hint for an alert, or None.

    Only when execution is enabled (IBK-126) — the hint would be noise (and
    /execute would just reject) while the bot is analysis-only.
    """
    if not context.settings.execution.enabled:
        return None
    try:
        with context.engine.connect() as conn:
            row = conn.execute(
                select(strategy_scores.c.id)
                .where(strategy_scores.c.snapshot_id == snapshot_id)
                .where(strategy_scores.c.strategy == strategy)
            ).one_or_none()
    except Exception:  # noqa: BLE001 -- a hint must never break alert delivery
        log.exception("execute-hint lookup failed for snapshot %s", snapshot_id)
        return None
    return f"/execute {row.id}" if row is not None else None


async def dispatch_alert(
    context: DaemonContext,
    alert_id: int,
    snapshot_id: int,
    scored: ScoredStrategy,
) -> None:
    """Send one alert; update the row's status based on outcome."""
    view = _load_view_for_snapshot(context, snapshot_id)
    snapshot_ts = _load_snapshot_ts(context, snapshot_id)
    symbol = _load_symbol_for_alert(context, alert_id)
    suggestion = scored.suggestion
    executable = bool(
        getattr(suggestion, "defined_risk", False)
        and getattr(suggestion, "suggested_quantity", 0) > 0
    )
    text = format_alert_markdown(
        symbol=symbol,
        view=view,
        scored=scored,
        snapshot_ts=snapshot_ts,
        execute_hint=(
            execute_hint_for(context, snapshot_id, scored.strategy_name) if executable else None
        ),
    )
    try:
        msg_id = await context.telegram.send_message(text)
    except Exception as e:  # noqa: BLE001 -- Telegram failure modes are heterogeneous
        log.exception("alert %d send failed", alert_id)
        _mark_failed(context, alert_id, str(e))
        return
    _mark_sent(context, alert_id, msg_id)


async def sweep_retries(context: DaemonContext) -> int:
    """Re-dispatch failed/pending alerts whose backoff window has elapsed.

    Returns the count of retries attempted (succeeded or failed again).
    """
    now = datetime.now(UTC)
    with context.engine.connect() as conn:
        rows = conn.execute(
            select(alerts.c.id, alerts.c.symbol, alerts.c.strategy, alerts.c.score)
            .where(
                and_(
                    alerts.c.status.in_(("pending", "failed")),
                    or_(
                        alerts.c.next_retry_ts.is_(None),
                        alerts.c.next_retry_ts <= now,
                    ),
                )
            )
            .order_by(alerts.c.id)
        ).fetchall()

    count = 0
    for row in rows:
        scored = _reconstruct_scored(context, row.id, row.strategy, row.score)
        snapshot_id = _snapshot_id_for_alert(context, row.id, row.symbol)
        if scored is None or snapshot_id is None:
            log.warning("retry skipped for alert %d: missing scored/snapshot", row.id)
            continue
        await dispatch_alert(context, row.id, snapshot_id, scored)
        count += 1
    return count


# ---- internals -------------------------------------------------------------


def _mark_sent(context: DaemonContext, alert_id: int, msg_id: int) -> None:
    now = datetime.now(UTC)
    with context.engine.begin() as conn:
        conn.execute(
            update(alerts)
            .where(alerts.c.id == alert_id)
            .values(
                status="sent",
                sent_ts=now,
                telegram_msg_id=msg_id,
                last_error=None,
                next_retry_ts=None,
            )
        )


def _mark_failed(context: DaemonContext, alert_id: int, error: str) -> None:
    with context.engine.connect() as conn:
        row = conn.execute(select(alerts.c.retry_count).where(alerts.c.id == alert_id)).fetchone()
    current = int(row.retry_count) if row else 0
    new_count = current + 1
    new_status: str
    next_retry_ts: datetime | None
    if new_count > MAX_RETRIES:
        new_status = "dropped"
        next_retry_ts = None
    else:
        new_status = "failed"
        backoff = BACKOFF_MINUTES[new_count - 1]
        next_retry_ts = datetime.now(UTC) + timedelta(minutes=backoff)
    with context.engine.begin() as conn:
        conn.execute(
            update(alerts)
            .where(alerts.c.id == alert_id)
            .values(
                status=new_status,
                retry_count=new_count,
                last_error=error,
                next_retry_ts=next_retry_ts,
            )
        )


def _load_view_for_snapshot(context: DaemonContext, snapshot_id: int) -> MarketView:
    with context.engine.connect() as conn:
        row = conn.execute(select(snapshots).where(snapshots.c.id == snapshot_id)).fetchone()
    if row is None:
        raise RuntimeError(f"snapshot {snapshot_id} not found")
    return MarketView(
        direction=cast("Direction", row.regime_dir) if row.regime_dir else "neutral",
        # direction_strength and earnings_in_window are not persisted on the
        # snapshots table. The formatter only reads direction/iv_regime/
        # iv_rank_value, so the values chosen here never reach the rendered
        # alert. Defaults kept simple.
        direction_strength="strong",
        iv_regime=cast("IVRegime", row.regime_iv) if row.regime_iv else "neutral",
        iv_rank_value=row.iv_rank,
        earnings_in_window=False,
        warming_up=(row.raw_json or {}).get("warming_up", False) if row.raw_json else False,
    )


def _load_snapshot_ts(context: DaemonContext, snapshot_id: int) -> datetime:
    with context.engine.connect() as conn:
        row = conn.execute(select(snapshots.c.ts).where(snapshots.c.id == snapshot_id)).fetchone()
    if row is None or row.ts is None:
        return datetime.now(UTC)
    ts: datetime = row.ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


def _load_symbol_for_alert(context: DaemonContext, alert_id: int) -> str:
    with context.engine.connect() as conn:
        row = conn.execute(select(alerts.c.symbol).where(alerts.c.id == alert_id)).fetchone()
    if row is None:
        raise RuntimeError(f"alert {alert_id} not found")
    return cast(str, row.symbol)


def _snapshot_id_for_alert(context: DaemonContext, alert_id: int, _symbol: str) -> int | None:
    with context.engine.connect() as conn:
        row = conn.execute(
            select(strategy_scores.c.snapshot_id)
            .select_from(
                alerts.join(
                    strategy_scores,
                    alerts.c.strategy_score_id == strategy_scores.c.id,
                )
            )
            .where(alerts.c.id == alert_id)
        ).first()
    return cast(int, row.snapshot_id) if row is not None else None


def _reconstruct_scored(
    context: DaemonContext,
    alert_id: int,
    strategy: str,
    score: float,
) -> ScoredStrategy | None:
    """Look up the persisted strategy_scores row for this alert's strategy and
    rebuild a minimal ScoredStrategy so format_alert_markdown can render it.

    Uses SimpleNamespace for the suggestion since the formatter only reads
    duck-typed fields (legs/credit_or_debit/max_loss/etc.).
    """
    with context.engine.connect() as conn:
        alert_row = conn.execute(
            select(alerts.c.strategy_score_id).where(alerts.c.id == alert_id)
        ).first()
        linked_score_id = (
            int(alert_row.strategy_score_id)
            if alert_row is not None and alert_row.strategy_score_id is not None
            else None
        )
        if linked_score_id is None:
            return None
        score_row = conn.execute(
            select(strategy_scores).where(strategy_scores.c.id == linked_score_id)
        ).fetchone()
    if score_row is None:
        return None
    legs_data = score_row.legs_json or []
    legs = tuple(
        SimpleNamespace(
            symbol=leg.get("symbol", ""),
            side=leg.get("side", ""),
            sec_type=leg.get("sec_type", "OPT"),
            expiry=leg.get("expiry"),
            strike=leg.get("strike"),
            right=leg.get("right"),
            quantity=leg.get("quantity", 1),
        )
        for leg in legs_data
    )
    # Pull the suggestion fields back from suggestion_json. The fallback
    # (defined_risk=True, financials None/0) only fires for legacy rows
    # written before migration 0002 added the column -- post-migration
    # retry alerts render exactly the same UNDEFINED RISK warning + figures
    # as the first attempt.
    sug_data = score_row.suggestion_json or {}
    sug = SimpleNamespace(
        legs=legs,
        credit_or_debit=sug_data.get("credit_or_debit", 0.0),
        max_loss=sug_data.get("max_loss"),
        max_profit=sug_data.get("max_profit"),
        prob_profit=sug_data.get("prob_profit"),
        suggested_quantity=sug_data.get("suggested_quantity", 0),
        defined_risk=sug_data.get("defined_risk", True),
        reward_risk=sug_data.get("reward_risk"),
        expected_value=sug_data.get("expected_value"),
        risk_tier=sug_data.get("risk_tier", "balanced"),
    )
    return ScoredStrategy(
        strategy_name=strategy,
        score=float(score_row.score),
        factors=FactorBreakdown(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),  # not persisted
        suggestion=sug,  # type: ignore[arg-type]  # duck-typed for formatter
        rationale=score_row.rationale or "",
    )
