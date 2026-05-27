"""Daemon lifecycle: build context, start scheduler, run until interrupted."""

from __future__ import annotations

import asyncio
import logging
import signal

from optionsbot.config import Settings, get_settings
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.scheduler import build_scheduler
from optionsbot.daemon.telegram_client import TelegramClient
from optionsbot.ibkr import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.storage.db import create_engine_for_path

log = logging.getLogger(__name__)


class Daemon:
    """Top-level daemon coordinator."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._context: DaemonContext | None = None
        self._stop_event = asyncio.Event()

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
            scheduler = build_scheduler(self._context, self._scan_tick)
            scheduler.start()
        except Exception:
            # build_scheduler / scheduler.start can raise (APScheduler
            # validates add_job args, executor configuration, etc.). Without
            # this guard a failure here would skip _shutdown_context and
            # leak the IBKR connection + Telegram client + engine.
            log.exception("Failed to start scheduler; daemon will exit")
            await self._shutdown_context()
            return 1

        log.info("Daemon started; waiting for stop signal")
        try:
            await self._stop_event.wait()
        finally:
            log.info("Stop signal received; shutting down scheduler")
            try:
                scheduler.shutdown(wait=True)
            except Exception:
                # SchedulerNotRunningError (or similar) must not prevent
                # IBKR disconnect + engine dispose; we always want a clean
                # process exit even if the scheduler self-terminated.
                log.exception("Scheduler shutdown failed")
            await self._shutdown_context()
        return 0

    def request_stop(self) -> None:
        """Set the stop event. Called by signal handlers (wired in Task 5)."""
        self._stop_event.set()

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
