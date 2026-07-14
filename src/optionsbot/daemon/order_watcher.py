"""1-minute order watcher (IBK-126): TTL sweep + terminal notifications.

Runs as an APScheduler job. No-ops fast when the daemon has no execution
wiring or no orders. Never raises — every order is handled in its own
try/except so one bad row can't starve the rest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

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


async def run_orders_tick(
    context: DaemonContext, *, now: datetime | None = None
) -> OrdersTickSummary:
    if context.order_client is None:
        return OrdersTickSummary(working=0, expired=0, notified=0)
    engine = context.engine
    now = now if now is not None else datetime.now(UTC)
    ttl = timedelta(minutes=context.settings.execution.order_ttl_minutes)

    # Work-stream D2: periodic broker reconciliation on a FIXED cadence —
    # regardless of whether open ledger rows exist. A forgotten broker position
    # leaves NO open order rows, so the old "only when open_orders" guard made
    # the position-compare unreachable exactly when it mattered. Startup always
    # runs one pass from the runner.
    rec_min = context.settings.execution.reconcile_minutes
    if rec_min > 0:
        last = context.last_reconcile_ts or context.started_at
        if now - last >= timedelta(minutes=rec_min):
            from optionsbot.execution.reconcile import reconcile

            async def _notify(text: str) -> None:
                await context.telegram.send_message(text, parse_mode=None)

            async def _positions() -> Any:
                from optionsbot.ibkr.positions import PositionsClient

                async with context.ibkr_lock:
                    return await PositionsClient(context.ibkr).get_portfolio()

            try:
                reconcile_summary = await reconcile(
                    engine, context.order_client, notify=_notify, now=now,
                    walk_md=_walk_md_for(context),
                    walk_tasks=context.walk_tasks,
                    settings=context.settings,
                    positions_snapshot=_positions,
                )
            except Exception as exc:
                from optionsbot.daemon.operational_state import record_reconcile_failure

                record_reconcile_failure(
                    phase="periodic", error_type=type(exc).__name__, now=now
                )
                raise
            from optionsbot.daemon.operational_state import record_reconcile

            record_reconcile(reconcile_summary, phase="periodic", now=now)
            if (
                (reconcile_summary.mismatches or reconcile_summary.orphan_positions)
                and context.events is not None
            ):
                context.events.emit(
                    "reconcile-mismatch",
                    "Periodic reconciliation found broker/ledger differences",
                    severity="critical",
                    details={
                        "mismatches": reconcile_summary.mismatches,
                        "orphan_positions": reconcile_summary.orphan_positions,
                    },
                )
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
        if row.status == "filled" and context.events is not None:
            context.events.emit(
                "fill",
                f"{row.intent} order #{row.id} filled: {row.symbol} {row.strategy} {row.quantity}x",
                details={
                    "order_id": row.id,
                    "intent": row.intent,
                    "symbol": row.symbol,
                    "strategy": row.strategy,
                    "quantity": row.quantity,
                    "net_premium": premium,
                },
            )
        row_ts = row.terminal_ts if row.terminal_ts.tzinfo else row.terminal_ts.replace(tzinfo=UTC)
        watermark = max(watermark, row_ts)
    context.orders_notified_through = watermark

    # IBK-130: a realized round-trip just completed — evaluate realized-only
    # loss triggers (consecutive-loss + realized daily loss). Backstop only.
    if close_filled:
        try:
            await _check_loss_kill_triggers(context, now)
        except Exception:  # noqa: BLE001 -- trigger evaluation must not kill the tick
            log.exception("loss kill-trigger evaluation failed")

    # PHASE 0 B1: net-liq (realized+unrealized) drawdown circuit breaker runs
    # EVERY tick, not just on a close fill — an open position bleeding past the
    # cap with nothing closed must still trip the kill.
    try:
        await _check_net_liq_drawdown(context, now)
    except Exception:  # noqa: BLE001 -- breaker must not kill the tick
        log.exception("net-liq drawdown evaluation failed")

    if expired or notified:
        log.info(
            "orders tick: working=%d expired=%d notified=%d",
            len(working), expired, notified,
        )
    return OrdersTickSummary(working=len(working), expired=expired, notified=notified)


async def _net_liq(context: DaemonContext) -> float | None:
    """Net liquidation in USD for the daily-loss / drawdown triggers.

    IBK-122: the daily-loss kill compares this against realized PnL, which is
    USD (option premiums), on an EUR-base account — so return the USD-converted
    net-liq, not the raw EUR base. The drawdown check (which also reads this) is
    a same-currency ratio, so USD keeps it self-consistent. Patchable seam for
    tests.
    """
    from optionsbot.ibkr.positions import PositionsClient

    async with context.ibkr_lock:
        summary = await PositionsClient(context.ibkr).get_account_summary()
    return (
        float(summary.net_liquidation_usd)
        if summary.net_liquidation_usd is not None
        else None
    )


async def _check_loss_kill_triggers(context: DaemonContext, now: datetime) -> None:
    """IBK-130: consecutive-loss and daily-realized-loss kill switches."""
    from optionsbot.execution.orders import realized_close_pairs
    from optionsbot.execution.state import load_state, trip_kill

    engine = context.engine
    if load_state(engine).killed:
        return

    from optionsbot.daemon.market_hours import nyse_session_start_utc

    session_start = nyse_session_start_utc(now)
    # PHASE 0 B3: scope BOTH the consecutive-loss streak and the realized-today
    # tally to THIS NYSE session. A global streak would let N stale losses from
    # a prior session trip the kill the instant one fresh close lands.
    session_pairs = realized_close_pairs(engine, since=session_start)
    if not session_pairs:
        return
    limit = context.settings.execution.max_consecutive_losses
    recent = session_pairs[-limit:]
    if len(recent) >= limit and all(p.pnl < 0 for p in recent):
        trip_kill(engine, f"{limit} consecutive losing trades this session")
        await _send(
            context,
            f"🛑 KILL SWITCH: {limit} consecutive losing trades this session "
            f"(latest {recent[-1].symbol} {recent[-1].pnl:+,.0f}). "
            "No new orders until /arm.",
        )
        return
    realized_today = sum(p.pnl for p in session_pairs)
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


async def _check_net_liq_drawdown(context: DaemonContext, now: datetime) -> None:
    """PHASE 0 B1: trip the kill on a realized+unrealized day-start drawdown."""
    from optionsbot.daemon.market_hours import nyse_session_date
    from optionsbot.execution.equity_guard import (
        capture_day_start_net_liq,
        evaluate_net_liq_drawdown,
    )
    from optionsbot.execution.state import load_state

    engine = context.engine
    if load_state(engine).killed:
        return
    net_liq = await _net_liq(context)
    if net_liq is None:
        return  # the realized backstop already warns when net-liq is unavailable
    # Capture the day-start baseline keyed by the NYSE session date so it resets
    # at the ET boundary but is idempotent (restart-stable) within a session.
    # context.day_start_net_liq mirrors it for /status without a DB hit.
    session = nyse_session_date(now).isoformat()
    context.day_start_net_liq = capture_day_start_net_liq(
        engine, net_liq, session=session
    )
    verdict = evaluate_net_liq_drawdown(
        engine, context.settings, current_net_liq=net_liq, now=now
    )
    if verdict.tripped:
        await _send(
            context,
            f"🛑 KILL SWITCH: {verdict.reason}. No new orders until /arm.",
        )


def _walk_md_for(context: DaemonContext) -> Any:
    if context.exec_ibkr is None:
        return None
    from optionsbot.ibkr.market_data import MarketDataClient
    return MarketDataClient(context.exec_ibkr)


async def _send(context: DaemonContext, text: str) -> None:
    try:
        await context.telegram.send_message(text, parse_mode=None)
    except Exception:  # noqa: BLE001 -- notification failure must not kill the sweep
        log.exception("order notification send failed")
