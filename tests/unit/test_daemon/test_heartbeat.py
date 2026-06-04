from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from sqlalchemy import insert

from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.runner import format_heartbeat
from optionsbot.storage.schema import scan_runs


def test_format_heartbeat_with_last_run() -> None:
    msg = format_heartbeat(scanned=7, alerts=3, finished=datetime(2026, 6, 4, 15, 59, tzinfo=UTC))
    assert "7" in msg and "3" in msg and "15:59" in msg


def test_format_heartbeat_no_runs() -> None:
    assert "no" in format_heartbeat(scanned=None, alerts=None, finished=None).lower()


async def test_heartbeat_tick_skips_when_market_closed(daemon_context: DaemonContext) -> None:
    from optionsbot.daemon.runner import Daemon
    d = Daemon(settings=daemon_context.settings)
    d._context = daemon_context
    daemon_context.telegram.send_message = AsyncMock()
    with patch("optionsbot.daemon.runner.is_market_open", return_value=False):
        await d._heartbeat_tick()
    daemon_context.telegram.send_message.assert_not_awaited()


async def test_heartbeat_tick_sends_when_market_open(daemon_context: DaemonContext) -> None:
    from optionsbot.daemon.runner import Daemon
    with daemon_context.engine.begin() as conn:
        conn.execute(insert(scan_runs).values(
            started=datetime(2026, 6, 4, 15, 56, tzinfo=UTC),
            finished=datetime(2026, 6, 4, 15, 59, tzinfo=UTC),
            tickers_scanned=7, alerts_fired=3,
        ))
    d = Daemon(settings=daemon_context.settings)
    d._context = daemon_context
    daemon_context.telegram.send_message = AsyncMock()
    with patch("optionsbot.daemon.runner.is_market_open", return_value=True):
        await d._heartbeat_tick()
    daemon_context.telegram.send_message.assert_awaited_once()
    assert daemon_context.telegram.send_message.await_args.kwargs.get("parse_mode") is None
