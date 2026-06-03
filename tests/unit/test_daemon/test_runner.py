"""Tests for the Daemon class: lifecycle + signal handlers (IBK-63)."""

from __future__ import annotations

import asyncio
import signal
from unittest.mock import AsyncMock, MagicMock, patch

from optionsbot.config import Settings
from optionsbot.daemon.runner import Daemon


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

    async def _ok_connect(self):
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


async def test_start_returns_1_on_ibkr_connect_failure(
    daemon_settings, monkeypatch,
) -> None:
    d = Daemon(settings=daemon_settings)

    async def _fail_connect(self):
        raise ConnectionError("gateway down")

    monkeypatch.setattr("optionsbot.ibkr.IBKRClient.connect", _fail_connect)
    code = await d.start()
    assert code == 1


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
