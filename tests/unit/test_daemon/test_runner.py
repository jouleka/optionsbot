"""Tests for the Daemon class: lifecycle + signal handlers (IBK-63)."""

from __future__ import annotations

import asyncio
import signal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from optionsbot.config import Settings
from optionsbot.daemon.runner import Daemon
from optionsbot.execution.reconcile import ReconcileSummary


async def test_request_stop_sets_event() -> None:
    d = Daemon()
    assert not d._stop_event.is_set()
    d.request_stop()
    assert d._stop_event.is_set()


async def test_signal_handlers_call_request_stop(daemon_settings) -> None:
    """install_signal_handlers wires SIGTERM and SIGINT to request_stop."""
    d = Daemon(settings=daemon_settings)
    loop = asyncio.get_event_loop()
    with patch.object(loop, "add_signal_handler") as mock_add:
        d.install_signal_handlers(loop)
    # Two handlers registered: SIGTERM and SIGINT.
    sigs = [call.args[0] for call in mock_add.call_args_list]
    assert signal.SIGTERM in sigs
    assert signal.SIGINT in sigs


async def test_start_exits_cleanly_on_stop_event(
    daemon_settings, monkeypatch,
) -> None:
    """start() returns 0 when stop_event is set, after shutting down cleanly."""
    d = Daemon(settings=daemon_settings)

    async def _ok_connect(self, *, forever: bool = False):
        return None

    monkeypatch.setattr("optionsbot.ibkr.IBKRClient.connect", _ok_connect)
    monkeypatch.setattr(
        "optionsbot.ibkr.IBKRClient.disconnect", AsyncMock()
    )

    async def _stop_soon() -> None:
        await asyncio.sleep(0.01)
        d.request_stop()

    asyncio.create_task(_stop_soon())
    code = await d.start()
    assert code == 0


async def test_start_returns_1_on_non_connection_connect_error(
    daemon_settings, monkeypatch,
) -> None:
    """A genuine (non-connection) error from connect still surfaces as exit 1.
    IBK-137: forever=True never raises on a CONNECTION failure (that path now
    WAITS -- see test_start_aborts_cleanly_when_stopped_before_connect), but a
    real programming error must still propagate to a clean exit-1."""
    d = Daemon(settings=daemon_settings)

    async def _raise_runtime(self, *, forever: bool = False):
        raise RuntimeError("boom")

    monkeypatch.setattr("optionsbot.ibkr.IBKRClient.connect", _raise_runtime)
    monkeypatch.setattr("optionsbot.ibkr.IBKRClient.disconnect", AsyncMock())
    code = await d.start()
    assert code == 1


async def test_start_aborts_cleanly_when_stopped_before_connect(
    daemon_settings, monkeypatch,
) -> None:
    """IBK-137: with forever=True a down Gateway makes connect WAIT; a stop
    signal during that wait must exit PROMPTLY with code 0 -- not crash-loop,
    and not stall until SIGKILL (the redeploy-during-outage path)."""
    d = Daemon(settings=daemon_settings)

    async def _never_connects(self, *, forever: bool = False):
        await asyncio.sleep(3600)  # Gateway never comes up

    monkeypatch.setattr("optionsbot.ibkr.IBKRClient.connect", _never_connects)
    monkeypatch.setattr("optionsbot.ibkr.IBKRClient.disconnect", AsyncMock())

    async def _stop_soon() -> None:
        await asyncio.sleep(0.01)
        d.request_stop()

    asyncio.create_task(_stop_soon())
    code = await d.start()
    assert code == 0


async def test_scan_tick_feeds_summary_to_gateway_health_paging(
    daemon_settings,
) -> None:
    """IBK-137 Inc 2: after each scan the tick hands the summary to the gateway
    health pager (wedge/disconnect detection). Light test: no Daemon.start(),
    all sibling ticks stubbed."""
    d = Daemon(settings=daemon_settings)
    d._context = MagicMock()
    fake_summary = MagicMock()
    fake_summary.tickers_scanned = 0
    fake_summary.alerts_enqueued = 0
    fake_summary.errors = ["SPY: TimeoutError (scan budget)"]

    with patch(
        "optionsbot.daemon.scan_runner.run_scan_tick",
        new=AsyncMock(return_value=fake_summary),
    ), patch(
        "optionsbot.daemon.gateway_health.page_gateway_health", new=AsyncMock()
    ) as mock_page, patch(
        "optionsbot.daemon.manage_runner.run_manage_tick",
        new=AsyncMock(return_value=MagicMock(positions_seen=0, alerts_sent=0, errors=[])),
    ), patch(
        "optionsbot.daemon.exit_runner.run_exits_tick", new=AsyncMock()
    ):
        await d._scan_tick()

    mock_page.assert_awaited_once()
    assert mock_page.await_args.args[1] is fake_summary  # (context, summary)


def test_config_summary_includes_key_fields(daemon_settings) -> None:
    from optionsbot.daemon.runner import _config_summary

    daemon_settings.scan.score_threshold = 65
    daemon_settings.scan.interval_minutes = 12
    s = _config_summary(daemon_settings)
    assert "telegram_configured=True" in s
    assert "threshold=65" in s
    assert "interval_min=12" in s


async def test_signal_handlers_register_sighup(daemon_settings) -> None:
    d = Daemon(settings=daemon_settings)
    loop = asyncio.get_event_loop()
    with patch.object(loop, "add_signal_handler") as mock_add:
        d.install_signal_handlers(loop)
    hup_calls = [c for c in mock_add.call_args_list if c.args[0] == signal.SIGHUP]
    assert len(hup_calls) == 1
    assert hup_calls[0].args[1] == d.request_reload


def test_request_reload_noop_without_context(daemon_settings) -> None:
    d = Daemon(settings=daemon_settings)
    assert d._context is None
    d.request_reload()  # must not raise (no running loop / no context)


async def test_reload_config_applies_new_settings(
    daemon_settings, daemon_context, monkeypatch
) -> None:
    d = Daemon(settings=daemon_settings)
    d._context = daemon_context
    d._scheduler = MagicMock()

    new = Settings()
    new.telegram.bot_token = "new-token"
    new.telegram.chat_id = "new-chat"
    new.scan.interval_minutes = 9
    monkeypatch.setattr("optionsbot.daemon.runner.load_settings", lambda: new)

    new_tg = MagicMock()
    new_tg.aclose = AsyncMock()
    monkeypatch.setattr(
        "optionsbot.daemon.runner.TelegramClient", MagicMock(return_value=new_tg)
    )
    old_tg = daemon_context.telegram

    await d._reload_config()

    assert d._settings is new
    assert d._context.settings is new
    assert d._context.telegram is new_tg
    old_tg.aclose.assert_awaited_once()
    d._scheduler.reschedule_job.assert_called_once()
    _, kwargs = d._scheduler.reschedule_job.call_args
    assert kwargs["trigger"].interval.total_seconds() == 9 * 60


async def test_request_reload_retains_task_and_guards_reentrancy(
    daemon_settings, daemon_context, monkeypatch
) -> None:
    """request_reload keeps a strong ref to the task (no GC) and a second call
    while one is in flight does not spawn a duplicate reload."""
    d = Daemon(settings=daemon_settings)
    d._context = daemon_context
    d._scheduler = MagicMock()
    release = asyncio.Event()

    async def _blocking_reload() -> None:
        await release.wait()

    monkeypatch.setattr(d, "_reload_config", _blocking_reload)

    d.request_reload()
    task1 = d._reload_task
    assert task1 is not None
    assert not task1.done()

    d.request_reload()  # reentrant call while task1 is still pending
    assert d._reload_task is task1  # no duplicate task spawned

    release.set()
    await task1
    assert task1.done()


async def test_startup_reconcile_passes_positions_snapshot_with_zero_open_rows(
    daemon_settings: Settings,
    daemon_context: Any,
    monkeypatch: Any,
) -> None:
    """IBK-136 I2: startup reconcile must call reconcile() with a non-None
    positions_snapshot even when the ledger has ZERO open rows, so the
    position-level orphan compare (Task 9) fires at boot and a fresh-start
    orphan position trips the kill rather than waiting for the periodic pass.

    This test FAILS against the pre-fix code because the old guard
    ``if _open_rows(self._context.engine):`` skipped the reconcile entirely
    when there were no open ledger rows.
    """
    d = Daemon(settings=daemon_settings)

    # Wire connect/disconnect so start() doesn't error before the reconcile.
    async def _ok_connect(self: Any) -> None:
        return None

    monkeypatch.setattr("optionsbot.ibkr.IBKRClient.connect", _ok_connect)
    monkeypatch.setattr("optionsbot.ibkr.IBKRClient.disconnect", AsyncMock())

    # Inject a mock order_client so the startup reconcile block is entered.
    daemon_context.order_client = MagicMock()

    # Capture the kwargs reconcile() is called with.
    captured: list[dict[str, Any]] = []

    async def fake_reconcile(engine: Any, client: Any, **kwargs: Any) -> ReconcileSummary:
        captured.append(kwargs)
        return ReconcileSummary(0, 0, 0, 0, 0, 0)

    monkeypatch.setattr("optionsbot.execution.reconcile.reconcile", fake_reconcile)

    # Confirm there are genuinely ZERO open ledger rows (ledger is freshly
    # migrated via daemon_engine fixture).
    from optionsbot.execution.orders import open_orders
    assert open_orders(daemon_context.engine) == []

    # Patch _build_context to return the pre-built daemon_context so the
    # Daemon uses our engine (with zero rows) and order_client.
    monkeypatch.setattr(d, "_build_context", lambda: daemon_context)

    async def _stop_soon() -> None:
        await asyncio.sleep(0.02)
        d.request_stop()

    asyncio.create_task(_stop_soon())
    code = await d.start()
    assert code == 0

    # Discriminating assertions:
    # 1. reconcile() was called at all (failed with the old ``if open_rows`` guard).
    assert len(captured) == 1, "startup reconcile must call reconcile() unconditionally"
    # 2. positions_snapshot was passed and is not None (so Task-9 orphan compare runs).
    assert "positions_snapshot" in captured[0], (
        "startup reconcile must pass positions_snapshot= to reconcile()"
    )
    assert captured[0]["positions_snapshot"] is not None, (
        "positions_snapshot must be a non-None callable, not omitted"
    )
