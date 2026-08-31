"""Daemon lifecycle: build context, start scheduler, run until interrupted."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from datetime import UTC, datetime, timedelta
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select

from optionsbot.config import Settings, get_settings, load_settings
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.market_hours import is_market_open
from optionsbot.daemon.scheduler import build_scheduler
from optionsbot.daemon.telegram_client import TelegramClient
from optionsbot.daemon.telegram_poller import poll_commands
from optionsbot.ibkr import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.storage.db import create_engine_for_path
from optionsbot.storage.schema import scan_runs

log = logging.getLogger(__name__)


def managed_capture_cron_seconds(interval: int, offset: int) -> str:
    """Cron second field for a research pass offset from protective exits."""
    return ",".join(str(second) for second in range(offset, 60, interval))


def format_heartbeat(scanned: int | None, alerts: int | None, finished: datetime | None) -> str:
    if finished is None:
        return "✅ optionsbot alive — no scan ticks yet"
    return f"✅ optionsbot alive — last tick {finished:%H:%M}Z: scanned {scanned}, {alerts} alerts"


def _config_summary(settings: Settings) -> str:
    """One-line, log-safe summary of the settings a running daemon is using."""
    configured = bool(settings.telegram.bot_token and settings.telegram.chat_id)
    return (
        f"telegram_configured={configured} "
        f"threshold={settings.scan.score_threshold} "
        f"interval_min={settings.scan.interval_minutes}"
    )


class Daemon:
    """Top-level daemon coordinator."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._context: DaemonContext | None = None
        self._stop_event = asyncio.Event()
        self._scheduler: AsyncIOScheduler | None = None
        self._reload_task: asyncio.Task[None] | None = None
        self._poller_task: asyncio.Task[None] | None = None

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Wire SIGTERM and SIGINT to request_stop() so Ctrl-C / systemd-stop
        triggers graceful shutdown instead of a noisy KeyboardInterrupt."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except NotImplementedError:
                # add_signal_handler isn't supported on Windows; the daemon
                # runs in WSL Ubuntu so this is the unusual path.
                pass
        # SIGHUP -> hot-reload config without a restart. Unix-only; getattr keeps
        # this import-safe on platforms (Windows) where SIGHUP doesn't exist.
        sighup = getattr(signal, "SIGHUP", None)
        if sighup is not None:
            try:
                loop.add_signal_handler(sighup, self.request_reload)
            except NotImplementedError:
                pass

    async def start(self) -> int:
        """Build context, start scheduler, run until stop_event is set. Returns exit code."""
        loop = asyncio.get_running_loop()
        self.install_signal_handlers(loop)
        self._context = self._build_context()
        # IBK-137: forever=True -> a down/wedged Gateway makes the daemon WAIT
        # (backoff-reconnect indefinitely) instead of raising -> exiting ->
        # systemd restart-looping. Race the wait against the stop signal so a
        # `systemctl stop/restart` DURING a Gateway outage exits promptly rather
        # than stalling until SIGKILL.
        connect_task = asyncio.create_task(self._context.ibkr.connect(forever=True))
        stop_task = asyncio.create_task(self._stop_event.wait())
        await asyncio.wait({connect_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if not connect_task.done():
            # Stop won the race (Gateway never came up): abort startup cleanly.
            connect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await connect_task
            log.info("Stop requested before IB Gateway connected; daemon exiting")
            await self._shutdown_context()
            return 0
        stop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stop_task
        try:
            connect_task.result()  # re-raise a genuine (non-connection) connect error
        except Exception:
            log.exception("Failed to connect to IB Gateway; daemon will exit")
            await self._shutdown_context()
            return 1

        from optionsbot.daemon.operational_state import record_daemon_started

        record_daemon_started()

        try:
            self._scheduler = build_scheduler(self._context, self._scan_tick)
            self._scheduler.start()
            self._register_periodic_jobs()
            # Created last so a failure registering the periodic jobs above can't
            # orphan the poller task (it wouldn't be cancelled in the except path).
            self._poller_task = asyncio.create_task(poll_commands(self._context))
        except Exception:
            log.exception("Failed to start scheduler; daemon will exit")
            await self._shutdown_context()
            return 1

        # IBK-128: startup reconciliation — adopt working orders across the
        # restart, resolve crash-orphaned rows, replay missed fills, and check
        # for orphan broker positions (IBK-136: I2 fix — run unconditionally so
        # the position-level compare fires even with zero open ledger rows).
        # Failure must never block boot (the periodic pass retries).
        if self._context.order_client is not None:
            try:
                from optionsbot.execution.reconcile import reconcile

                telegram = self._context.telegram

                async def _notify(text: str) -> None:
                    await telegram.send_message(text, parse_mode=None)

                from optionsbot.daemon.order_watcher import _walk_md_for

                async def _positions() -> Any:
                    from optionsbot.ibkr.positions import PositionsClient

                    async with self._context.ibkr_lock:  # type: ignore[union-attr]
                        return await PositionsClient(self._context.ibkr).get_portfolio()  # type: ignore[union-attr]

                summary = await reconcile(
                    self._context.engine,
                    self._context.order_client,
                    notify=_notify,
                    walk_md=_walk_md_for(self._context),
                    walk_tasks=self._context.walk_tasks,
                    walk_lock=self._context.ibkr_lock,
                    settings=self._settings,
                    positions_snapshot=_positions,
                )
                log.info(
                    "startup reconcile: adopted=%d foreign=%d fills=%d "
                    "resolved=%d mismatches=%d orphan_positions=%d",
                    summary.adopted,
                    summary.foreign,
                    summary.fills_replayed,
                    summary.resolved,
                    summary.mismatches,
                    summary.orphan_positions,
                )
                from optionsbot.daemon.operational_state import record_reconcile

                record_reconcile(summary, phase="startup")
                if (
                    summary.mismatches or summary.orphan_positions
                ) and self._context.events is not None:
                    self._context.events.emit(
                        "reconcile-mismatch",
                        "Startup reconciliation found broker/ledger differences",
                        severity="critical",
                        details={
                            "mismatches": summary.mismatches,
                            "orphan_positions": summary.orphan_positions,
                        },
                    )
                self._context.last_reconcile_ts = datetime.now(UTC)
            except Exception as exc:
                from optionsbot.daemon.operational_state import record_reconcile_failure

                record_reconcile_failure(phase="startup", error_type=type(exc).__name__)
                log.exception("startup reconciliation failed; periodic pass will retry")

        # Evaluate persisted Hermes evidence before the daemon advertises
        # readiness. This is separate from broker reconciliation and can only
        # disable Hermes-vetted entries, never exits or deterministic scans.
        await self._overlay_guard_tick()

        log.info("Daemon started; waiting for stop signal")
        log.info("daemon config: %s", _config_summary(self._settings))
        try:
            await self._stop_event.wait()
        finally:
            log.info("Stop signal received; shutting down scheduler")
            try:
                self._scheduler.shutdown(wait=True)
            except Exception:
                # SchedulerNotRunningError (or similar) must not prevent
                # IBKR disconnect + engine dispose; we always want a clean
                # process exit even if the scheduler self-terminated.
                log.exception("Scheduler shutdown failed")
            if self._poller_task is not None:
                self._poller_task.cancel()
                try:
                    await self._poller_task
                except asyncio.CancelledError:
                    pass
            await self._shutdown_context()
        return 0

    def request_stop(self) -> None:
        """Set the stop event. Called by signal handlers (wired in Task 5)."""
        self._stop_event.set()

    def request_reload(self) -> None:
        """Schedule an async config reload (wired to SIGHUP). No-op pre-start.

        Retains a strong reference to the task (asyncio only holds a weak one,
        so a fire-and-forget task can be GC'd mid-flight), and skips if a reload
        is already running so two SIGHUPs in quick succession don't overlap.
        """
        if self._context is None or self._scheduler is None:
            return
        if self._reload_task is not None and not self._reload_task.done():
            return
        self._reload_task = asyncio.create_task(self._reload_config())

    async def _reload_config(self) -> None:
        """Re-read settings and apply them live: telegram client, context
        settings, and the scan interval. The IBKR connection + engine are NOT
        reloaded (host/port/db changes still need a restart)."""
        if self._context is None or self._scheduler is None:
            return
        get_settings.cache_clear()
        new = load_settings()
        self._settings = new
        self._context.settings = new
        old_events = self._context.events
        from optionsbot.daemon.event_webhook import EventWebhookPublisher

        self._context.events = EventWebhookPublisher(new.hermes_webhook)
        if old_events is not None:
            await old_events.flush()
        old_telegram = self._context.telegram
        self._context.telegram = TelegramClient(new.telegram.bot_token, new.telegram.chat_id)
        try:
            await old_telegram.aclose()
        except Exception:  # noqa: BLE001 -- closing the old client must not abort reload
            log.exception("closing old Telegram client during reload failed")
        self._scheduler.reschedule_job(
            "scan", trigger=IntervalTrigger(minutes=new.scan.interval_minutes)
        )
        self._sync_managed_capture_job()
        self._sync_managed_learning_job()
        log.info("config reloaded: %s", _config_summary(new))

    def _build_context(self) -> DaemonContext:
        from optionsbot.daemon.event_webhook import EventWebhookPublisher
        from optionsbot.execution.tracker import OrderTracker
        from optionsbot.ibkr.orders import OrderClient

        engine = create_engine_for_path(self._settings.storage.db_path)
        ibkr = IBKRClient(role="daemon", settings=self._settings)
        resolver = ContractResolver(ibkr)
        telegram = TelegramClient(
            self._settings.telegram.bot_token, self._settings.telegram.chat_id
        )
        # IBK-126: execution plumbing. Constructed unconditionally (cheap — no
        # connection until the first order op); the gate keeps it inert while
        # execution.enabled is false. Dedicated clientId 3 connection because
        # order events only reach the placing clientId. Shares the resolver so
        # leg qualification reuses the scan's contract cache.
        exec_ibkr = IBKRClient(role="exec", settings=self._settings)
        order_client = OrderClient(exec_ibkr, resolver)
        OrderTracker(engine).attach(order_client)
        events = EventWebhookPublisher(self._settings.hermes_webhook)
        return DaemonContext(
            settings=self._settings,
            engine=engine,
            ibkr=ibkr,
            resolver=resolver,
            telegram=telegram,
            events=events,
            exec_ibkr=exec_ibkr,
            order_client=order_client,
        )

    async def _shutdown_context(self) -> None:
        if self._context is None:
            return
        if self._context.events is not None:
            await self._context.events.flush()
        try:
            await self._context.ibkr.disconnect()
        except Exception:
            log.exception("IBKR disconnect failed")
        if self._context.exec_ibkr is not None:
            try:
                await self._context.exec_ibkr.disconnect()
            except Exception:
                log.exception("exec IBKR disconnect failed")
        try:
            await self._context.telegram.aclose()
        except Exception:
            log.exception("Telegram client close failed")
        self._context.engine.dispose()

    async def _scan_tick(self) -> None:
        from optionsbot.daemon.manage_runner import run_manage_tick
        from optionsbot.daemon.scan_runner import run_scan_tick

        assert self._context is not None
        try:
            summary = await run_scan_tick(self._context)
            log.info(
                "scan tick: scanned=%d alerts=%d errors=%d",
                summary.tickers_scanned,
                summary.alerts_enqueued,
                len(summary.errors),
            )
            # IBK-137 Inc 2: page the human on gateway wedge/disconnect. The
            # helper never raises; called from the source module so tests can
            # patch optionsbot.daemon.gateway_health.page_gateway_health.
            from optionsbot.daemon import gateway_health

            await gateway_health.page_gateway_health(self._context, summary)
        except Exception:
            log.exception("scan tick failed catastrophically")
        # Position-management pass runs as a sibling of the scan; its own try/except
        # so a management failure can't poison the scan (and vice-versa).
        try:
            ms = await run_manage_tick(self._context)
            log.info(
                "manage tick: positions=%d alerts=%d errors=%d",
                ms.positions_seen,
                ms.alerts_sent,
                len(ms.errors),
            )
        except Exception:
            log.exception("manage tick failed catastrophically")
        # Protective exits run on their own wall-clock-aligned sub-minute job.
        # Keeping them out of the scanner prevents scan duration/restarts from
        # shifting the 0DTE force-exit deadline.

    async def _exits_tick(self) -> None:
        """Run protective exits independently of scan latency and cadence."""
        assert self._context is not None
        try:
            from optionsbot.daemon.exit_runner import run_exits_tick

            await run_exits_tick(self._context)
        except Exception:
            log.exception("exits tick failed catastrophically")

    async def _managed_capture_tick(self) -> None:
        """Run bounded shadow quote capture; never place or manage an order."""
        assert self._context is not None
        try:
            from optionsbot.daemon.managed_capture import run_managed_capture_tick

            summary = await run_managed_capture_tick(self._context)
            if summary.opportunities_seen or summary.skipped_for_trading:
                log.info(
                    "managed capture tick: opportunities=%d usable=%d unusable=%d "
                    "resolved=%d censored=%d trading_priority_skip=%s quote_errors=%d",
                    summary.opportunities_seen,
                    summary.usable_marks,
                    summary.unusable_marks,
                    summary.resolved,
                    summary.censored,
                    summary.skipped_for_trading,
                    summary.quote_errors,
                )
        except Exception:
            # Shadow research must not poison exits, scanning, or daemon life.
            log.exception("managed capture tick failed")

    async def _managed_learning_tick(self) -> None:
        """Train/evaluate immutable challengers off-loop after market hours."""
        assert self._context is not None
        try:
            from optionsbot.daemon.managed_learning import run_managed_learning_tick

            summary = await run_managed_learning_tick(self._context)
            if summary.status not in {"market_open", "already_registered"}:
                log.info(
                    "managed learning: status=%s samples=%d sessions=%d "
                    "base_eligible=%s context_eligible=%s incremental=%s promoted=%s",
                    summary.status,
                    summary.samples,
                    summary.sessions,
                    summary.base_eligible,
                    summary.context_eligible,
                    summary.context_incremental_eligible,
                    summary.promoted_model_version,
                )
        except Exception:
            # Learning is never allowed to impair scanning, orders, or exits.
            log.exception("managed learning tick failed")

    async def _heartbeat_tick(self) -> None:
        assert self._context is not None
        if not is_market_open(datetime.now(UTC)):
            return
        with self._context.engine.connect() as conn:
            last = conn.execute(select(scan_runs).order_by(scan_runs.c.id.desc()).limit(1)).first()
        if last is None:
            msg = format_heartbeat(None, None, None)
        else:
            msg = format_heartbeat(last.tickers_scanned, last.alerts_fired, last.finished)
        from optionsbot.execution.state import load_state

        execution_state = load_state(self._context.engine)
        if execution_state.killed:
            msg += f"\n🛑 execution HALTED — {execution_state.reason or 'no reason recorded'}"
        elif self._context.settings.execution.enabled:
            msg += "\n✅ execution armed"
        else:
            msg += "\n⚪ execution disabled by config"
        try:
            await self._context.telegram.send_message(msg, parse_mode=None)
        except Exception:  # noqa: BLE001 -- heartbeat failure must not crash the daemon
            log.exception("heartbeat send failed")

    async def _outcomes_tick(self) -> None:
        from optionsbot.daemon.outcomes_runner import run_outcomes_tick

        assert self._context is not None
        async with self._context.hermes_overlay_lock:
            try:
                n = await run_outcomes_tick(self._context)
                log.info("outcomes tick: evaluated %d newly-expired pick(s)", n)
            except Exception:
                log.exception("outcomes tick failed")
            await self._evaluate_overlay_guard()

    async def _overlay_guard_tick(self) -> None:
        assert self._context is not None
        async with self._context.hermes_overlay_lock:
            await self._evaluate_overlay_guard()

    async def _evaluate_overlay_guard(self) -> None:
        """Persist and announce a newly-tripped Hermes correctness breaker."""
        assert self._context is not None
        try:
            from optionsbot.hermes_overlay import evaluate_overlay, hold_pending_reviews

            state, tripped = evaluate_overlay(self._context.engine)
            if not tripped:
                return
            held = hold_pending_reviews(self._context.engine, state)
            message = (
                "🛑 Hermes entry overlay DISABLED\n"
                f"{state.reason}\n"
                f"Held {held} pending review(s). Scans and exits continue. "
                "Use /overlayreset only after human review."
            )
            await self._context.telegram.send_message(message, parse_mode=None)
            if self._context.events is not None:
                self._context.events.emit(
                    "overlay-disabled",
                    state.reason or "Hermes entry overlay disabled",
                    severity="critical",
                    details={
                        "judgeable": state.judgeable,
                        "accuracy": state.accuracy,
                        "held_reviews": held,
                    },
                )
            log.critical("%s; held_reviews=%d", state.reason, held)
        except Exception:
            # A failed correctness evaluation must not affect exits/scans. The
            # entry path independently reads the last persisted breaker state.
            log.exception("Hermes overlay guard evaluation failed")

    async def _entry_reviews_tick(self) -> None:
        from optionsbot.daemon.auto_executor import run_entry_reviews_tick

        assert self._context is not None
        try:
            n = await run_entry_reviews_tick(self._context)
            if n:
                log.info("entry-review tick: submitted=%d", n)
        except Exception:
            log.exception("entry-review tick failed")

    async def _control_intents_tick(self) -> None:
        """Import bounded requests from the unprivileged Hermes queue."""
        path = os.environ.get("OPTIONSBOT_MCP_INTENT_DB_PATH")
        if not path:
            return
        assert self._context is not None
        try:
            from optionsbot.daemon.control_intents import consume_control_intents_async

            consumed = await consume_control_intents_async(self._context, path)
            if consumed:
                log.info("restricted MCP intents consumed=%d", consumed)
        except Exception:
            log.exception("restricted MCP intent consumer failed")

    async def _orders_tick(self) -> None:
        from optionsbot.daemon.order_watcher import run_orders_tick

        assert self._context is not None
        try:
            await run_orders_tick(self._context)
        except Exception:
            log.exception("orders tick failed")

    def _register_periodic_jobs(self) -> None:
        """Add the heartbeat + daily outcome-accrual jobs to the running scheduler. Each is
        gated by its config (telegram.heartbeat_minutes / validation.outcomes_eval_hours;
        0 disables)."""
        assert self._scheduler is not None
        hb = self._settings.telegram.heartbeat_minutes
        if hb > 0:
            self._scheduler.add_job(
                self._heartbeat_tick,
                trigger=IntervalTrigger(minutes=hb),
                id="heartbeat",
                max_instances=1,
                coalesce=True,
                replace_existing=True,
            )
        # IBK-117: daily outcome accrual (evaluate newly-expired picks). First run a couple
        # minutes after start so the ledger updates without waiting a full day.
        oeh = self._settings.validation.outcomes_eval_hours
        if oeh > 0:
            # Same-day learning cannot wait an hour (or the normal daily
            # cadence) after the bell: the EOD analyst needs settled call
            # outcomes before its report. The runner itself advances today's
            # evaluation date only after the official exchange close, so
            # pre-close 15-minute ticks are cheap no-ops and never score an
            # incomplete bar.
            outcomes_trigger = (
                IntervalTrigger(minutes=15)
                if self._settings.execution.zero_dte_only
                else IntervalTrigger(hours=oeh)
            )
            self._scheduler.add_job(
                self._outcomes_tick,
                trigger=outcomes_trigger,
                id="outcomes",
                max_instances=1,
                coalesce=True,
                replace_existing=True,
                next_run_time=datetime.now(UTC) + timedelta(minutes=2),
            )
        # IBK-126: order watcher — TTL sweep on working orders + terminal-state
        # Telegram notifications. Always registered (no-ops fast when there is
        # no execution wiring or no orders); NOT gated on execution.enabled
        # because working orders must still be managed after a /kill.
        self._scheduler.add_job(
            self._orders_tick,
            trigger=IntervalTrigger(minutes=1),
            id="orders",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        # Hermes entry reviews are advisory requests, never direct orders. This
        # consumer feeds fresh vetted requests back through execute_pick once a
        # minute; all deterministic execution gates remain authoritative.
        self._scheduler.add_job(
            self._entry_reviews_tick,
            trigger=IntervalTrigger(minutes=1),
            id="entry_reviews",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        # Protective exits are deadline-sensitive (especially 0DTE premium
        # stops). CronTrigger anchors them to stable wall-clock boundaries
        # instead of inheriting a daemon-start offset or scanner duration.
        exit_interval = self._settings.execution.exit_check_interval_seconds
        self._scheduler.add_job(
            self._exits_tick,
            trigger=CronTrigger(second=f"*/{exit_interval}", timezone=UTC),
            id="exits",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        self._sync_managed_capture_job()
        self._sync_managed_learning_job()
        # The MCP process cannot touch the trading DB. It appends only typed
        # intents to a separate queue; this trusted daemon translates them and
        # all downstream entry/exit gates independently revalidate the request.
        if os.environ.get("OPTIONSBOT_MCP_INTENT_DB_PATH"):
            self._scheduler.add_job(
                self._control_intents_tick,
                trigger=IntervalTrigger(seconds=10),
                id="control_intents",
                max_instances=1,
                coalesce=True,
                replace_existing=True,
            )

    def _sync_managed_capture_job(self) -> None:
        """Add/reschedule/remove the observational capture job after reloads."""
        assert self._scheduler is not None
        job_id = "managed_capture"
        validation = self._settings.validation
        existing = self._scheduler.get_job(job_id)
        if not validation.managed_capture_enabled:
            if existing is not None:
                self._scheduler.remove_job(job_id)
            return
        seconds = managed_capture_cron_seconds(
            validation.managed_capture_interval_seconds,
            validation.managed_capture_offset_seconds,
        )
        self._scheduler.add_job(
            self._managed_capture_tick,
            trigger=CronTrigger(second=seconds, timezone=UTC),
            id=job_id,
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )

    def _sync_managed_learning_job(self) -> None:
        """Add/reschedule/remove the CPU-bound post-session learner."""
        assert self._scheduler is not None
        job_id = "managed_learning"
        config = self._settings.managed_learning
        existing = self._scheduler.get_job(job_id)
        if not config.enabled:
            if existing is not None:
                self._scheduler.remove_job(job_id)
            return
        self._scheduler.add_job(
            self._managed_learning_tick,
            trigger=IntervalTrigger(hours=config.training_interval_hours),
            id=job_id,
            max_instances=1,
            coalesce=True,
            replace_existing=True,
            next_run_time=datetime.now(UTC) + timedelta(minutes=3),
        )
