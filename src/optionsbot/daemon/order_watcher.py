"""1-minute order watcher (IBK-126): TTL sweep + terminal notifications.

Runs as an APScheduler job. No-ops fast when the daemon has no execution
wiring or no orders. Never raises — every order is handled in its own
try/except so one bad row can't starve the rest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from optionsbot.daemon.context import DaemonContext
from optionsbot.execution.orders import (
    net_premium,
    set_order_note,
    working_orders,
)
from optionsbot.storage.schema import orders

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OrdersTickSummary:
    working: int
    expired: int
    notified: int


def _terminal_message(row: object, premium: float | None) -> str:
    status: str = row.status  # type: ignore[attr-defined]
    head = f"#{row.id} {row.symbol} {row.strategy} {row.quantity}x"  # type: ignore[attr-defined]
    err = row.last_error or ""  # type: ignore[attr-defined]
    if status == "filled":
        net = f" — net ${premium:,.0f}" if premium is not None else ""
        return f"🟢 filled {head}{net}"
    if status == "rejected":
        return f"🔴 rejected {head}: {err or 'no reason recorded'}"
    if status in ("cancelled", "abandoned"):
        if premium is not None:
            # Fills landed before the cancel: a REAL partial position exists
            # that the exit engine will not manage — human must flatten/adopt.
            return (
                f"⚠ PARTIAL FILL then {status}: {head} — net ${premium:,.0f} "
                "already executed. This partial position is NOT auto-managed; "
                "flatten it manually in TWS or ask Claude to adopt it."
            )
        if "walk exhausted" in err or "TTL" in err:
            return f"⏱ no fill {head} — trade skipped ({err})"
        return f"⚪ {status} {head}"
    if status == "skipped":
        return f"⚪ skipped {head}: {err or 'gates'}"
    return f"⚪ {status} {head}"


async def run_orders_tick(context: DaemonContext) -> OrdersTickSummary:
    if context.order_client is None:
        return OrdersTickSummary(working=0, expired=0, notified=0)
    engine = context.engine
    now = datetime.now(UTC)
    ttl = timedelta(minutes=context.settings.execution.order_ttl_minutes)

    # IBK-128: periodic broker reconciliation, only while non-terminal orders
    # exist (free when idle; startup always runs one pass from the runner).
    rec_min = context.settings.execution.reconcile_minutes
    if rec_min > 0:
        last = context.last_reconcile_ts or context.started_at
        if now - last >= timedelta(minutes=rec_min):
            from optionsbot.execution.orders import open_orders as open_rows
            from optionsbot.execution.reconcile import reconcile

            if open_rows(engine):
                async def _notify(text: str) -> None:
                    await context.telegram.send_message(text, parse_mode=None)

                await reconcile(engine, context.order_client, notify=_notify, now=now)
            context.last_reconcile_ts = now

    # --- TTL sweep: cancel at the broker FIRST, only then mark abandoned.
    expired = 0
    working = working_orders(engine)
    for order in working:
        if order.submitted_ts is None or now - order.submitted_ts <= ttl:
            continue
        if order.ib_order_id is None:
            continue  # never acked; reconciliation (IBK-128) owns this case
        try:
            set_order_note(
                engine, order.id,
                f"TTL {context.settings.execution.order_ttl_minutes}m expired "
                "unfilled — cancel requested",
            )
            await context.order_client.cancel(order.ib_order_id)
        except ValueError:
            # Placed before a daemon restart: the in-memory modify/cancel
            # registry is gone. Reconciliation re-adopts within minutes; warn
            # once meanwhile.
            if order.id not in context.orders_cancel_warned:
                context.orders_cancel_warned.add(order.id)
                await _send(
                    context,
                    f"⚠ order #{order.id} is working at IBKR but unmanaged "
                    "after a restart — reconciliation will adopt it shortly",
                )
            continue
        except Exception:  # noqa: BLE001 -- one bad cancel must not starve the sweep
            log.exception("TTL cancel failed for order %s", order.id)
            continue
        # NO terminal transition here: the broker owns the order's fate. The
        # tracker confirms (cancelled — or filled/partial if a fill raced us),
        # and this sweep simply retries while the row stays working.
        expired += 1

    # --- Notify newly-terminal orders exactly once (watermark on terminal_ts).
    since = context.orders_notified_through or context.started_at
    with engine.connect() as conn:
        rows = conn.execute(
            select(orders)
            .where(orders.c.terminal_ts.is_not(None))
            .where(orders.c.terminal_ts > since)
            .order_by(orders.c.terminal_ts)
        ).fetchall()
    notified = 0
    watermark = since
    close_filled = False
    for row in rows:
        premium = net_premium(engine, row.id) if row.status == "filled" else None
        await _send(context, _terminal_message(row, premium))
        notified += 1
        if row.status == "filled" and row.intent == "close":
            close_filled = True
        row_ts = row.terminal_ts if row.terminal_ts.tzinfo else row.terminal_ts.replace(tzinfo=UTC)
        watermark = max(watermark, row_ts)
    context.orders_notified_through = watermark

    # IBK-130: a realized round-trip just completed — evaluate loss triggers.
    if close_filled:
        try:
            await _check_loss_kill_triggers(context, now)
        except Exception:  # noqa: BLE001 -- trigger evaluation must not kill the tick
            log.exception("loss kill-trigger evaluation failed")

    if expired or notified:
        log.info(
            "orders tick: working=%d expired=%d notified=%d",
            len(working), expired, notified,
        )
    return OrdersTickSummary(working=len(working), expired=expired, notified=notified)


async def _net_liq(context: DaemonContext) -> float | None:
    """Net liquidation for the daily-loss trigger. Patchable seam for tests."""
    from optionsbot.ibkr.positions import PositionsClient

    async with context.ibkr_lock:
        summary = await PositionsClient(context.ibkr).get_account_summary()
    return (
        float(summary.net_liquidation)
        if summary.net_liquidation is not None
        else None
    )


async def _check_loss_kill_triggers(context: DaemonContext, now: datetime) -> None:
    """IBK-130: consecutive-loss and daily-realized-loss kill switches."""
    from optionsbot.execution.orders import realized_close_pairs
    from optionsbot.execution.state import load_state, trip_kill

    engine = context.engine
    if load_state(engine).killed:
        return
    pairs = realized_close_pairs(engine)
    if not pairs:
        return
    limit = context.settings.execution.max_consecutive_losses
    recent = pairs[-limit:]
    if len(recent) >= limit and all(p.pnl < 0 for p in recent):
        trip_kill(engine, f"{limit} consecutive losing trades")
        await _send(
            context,
            f"🛑 KILL SWITCH: {limit} consecutive losing trades "
            f"(latest {recent[-1].symbol} {recent[-1].pnl:+,.0f}). "
            "No new orders until /arm.",
        )
        return
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    realized_today = sum(
        p.pnl for p in pairs if p.closed_ts is not None and p.closed_ts >= day_start
    )
    if realized_today >= 0:
        return
    net_liq = await _net_liq(context)
    threshold = context.settings.execution.max_daily_loss_pct
    if net_liq is None:
        # Fail-open would silently hide a loss day (Opus IBK-130 #2) — say so.
        log.warning("daily-loss kill not evaluable: net liquidation unavailable")
        await _send(
            context,
            f"⚠ realized today ${realized_today:,.0f} but net liquidation is "
            "unavailable — the daily-loss kill switch could NOT be evaluated.",
        )
        return
    if abs(realized_today) >= threshold * net_liq:
        trip_kill(
            engine,
            f"daily realized loss ${abs(realized_today):,.0f} ≥ "
            f"{threshold * 100:.0f}% of net liq",
        )
        await _send(
            context,
            f"🛑 KILL SWITCH: daily loss ${abs(realized_today):,.0f} hit the "
            f"{threshold * 100:.0f}% cap. No new orders until /arm.",
        )


async def _send(context: DaemonContext, text: str) -> None:
    try:
        await context.telegram.send_message(text, parse_mode=None)
    except Exception:  # noqa: BLE001 -- notification failure must not kill the sweep
        log.exception("order notification send failed")
