"""Daemon lifecycle: build context, start scheduler, run until interrupted."""

from __future__ import annotations

import asyncio
import logging
import signal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from optionsbot.config import Settings, get_settings, load_settings
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.scheduler import build_scheduler
from optionsbot.daemon.telegram_client import TelegramClient
from optionsbot.daemon.telegram_poller import poll_commands
from optionsbot.ibkr import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.storage.db import create_engine_for_path

log = logging.getLogger(__name__)


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
        try:
            await self._context.ibkr.connect()
        except Exception:
            log.exception("Failed to connect to IB Gateway; daemon will exit")
            await self._shutdown_context()
            return 1

        try:
            self._scheduler = build_scheduler(self._context, self._scan_tick)
            self._scheduler.start()
            self._poller_task = asyncio.create_task(poll_commands(self._context))
        except Exception:
            log.exception("Failed to start scheduler; daemon will exit")
            await self._shutdown_context()
            return 1

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
        old_telegram = self._context.telegram
        self._context.telegram = TelegramClient(
            new.telegram.bot_token, new.telegram.chat_id
        )
        try:
            await old_telegram.aclose()
        except Exception:  # noqa: BLE001 -- closing the old client must not abort reload
            log.exception("closing old Telegram client during reload failed")
        self._scheduler.reschedule_job(
            "scan", trigger=IntervalTrigger(minutes=new.scan.interval_minutes)
        )
        log.info("config reloaded: %s", _config_summary(new))

    def _build_context(self) -> DaemonContext:
        engine = create_engine_for_path(self._settings.storage.db_path)
        ibkr = IBKRClient(role="daemon", settings=self._settings)
        resolver = ContractResolver(ibkr)
        telegram = TelegramClient(
            self._settings.telegram.bot_token, self._settings.telegram.chat_id
        )
        return DaemonContext(
            settings=self._settings,
            engine=engine,
            ibkr=ibkr,
            resolver=resolver,
            telegram=telegram,
        )

    async def _shutdown_context(self) -> None:
        if self._context is None:
            return
        try:
            await self._context.ibkr.disconnect()
        except Exception:
            log.exception("IBKR disconnect failed")
        try:
            await self._context.telegram.aclose()
        except Exception:
            log.exception("Telegram client close failed")
        self._context.engine.dispose()

    async def _scan_tick(self) -> None:
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
        except Exception:
            log.exception("scan tick failed catastrophically")
