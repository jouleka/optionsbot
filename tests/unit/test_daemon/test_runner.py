"""Tests for the Daemon class: lifecycle + signal handlers (IBK-63)."""

from __future__ import annotations

import asyncio
import signal
from unittest.mock import AsyncMock, patch

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
