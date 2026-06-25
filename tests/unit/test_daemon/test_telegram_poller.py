from __future__ import annotations

from unittest.mock import AsyncMock, patch

from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.telegram_poller import _initial_offset, poll_once

NOW_TS = 1_000_000.0


def _update(uid: int, chat_id: str, text: str, date: int | None = None) -> dict:
    message: dict = {"chat": {"id": int(chat_id)}, "text": text}
    if date is not None:
        message["date"] = date
    return {"update_id": uid, "message": message}


async def test_initial_offset_drops_stale_backlog(daemon_context: DaemonContext) -> None:
    # An old command in the backlog is dropped (offset advances past it) — keeps
    # the original "don't replay stale commands on startup" safety.
    updates = [_update(41, "5356256463", "/status", date=int(NOW_TS) - 3600)]
    daemon_context.telegram.get_updates = AsyncMock(return_value=updates)
    offset = await _initial_offset(daemon_context, now_ts=NOW_TS)
    assert offset == 42  # last+1 -> dropped


async def test_initial_offset_dateless_update_treated_stale(
    daemon_context: DaemonContext,
) -> None:
    # Defensive: an update with no date can't be confirmed fresh -> dropped.
    updates = [_update(41, "5356256463", "/status")]
    daemon_context.telegram.get_updates = AsyncMock(return_value=updates)
    offset = await _initial_offset(daemon_context, now_ts=NOW_TS)
    assert offset == 42


async def test_initial_offset_processes_fresh_backlog(
    daemon_context: DaemonContext,
) -> None:
    # THE FIX: a command sent moments before a restart (fresh) is NOT dropped —
    # the loop starts AT its update_id so poll_once dispatches it.
    updates = [_update(41, "5356256463", "/close 8", date=int(NOW_TS) - 5)]
    daemon_context.telegram.get_updates = AsyncMock(return_value=updates)
    offset = await _initial_offset(daemon_context, now_ts=NOW_TS)
    assert offset == 41  # start AT the fresh command, not 42


async def test_initial_offset_skips_stale_keeps_fresh(
    daemon_context: DaemonContext,
) -> None:
    # Mixed backlog (stale then fresh): start at the first fresh update.
    updates = [
        _update(40, "5356256463", "/status", date=int(NOW_TS) - 7200),  # stale
        _update(41, "5356256463", "/close 8", date=int(NOW_TS) - 10),   # fresh
    ]
    daemon_context.telegram.get_updates = AsyncMock(return_value=updates)
    offset = await _initial_offset(daemon_context, now_ts=NOW_TS)
    assert offset == 41


async def test_initial_offset_empty_backlog_returns_none(
    daemon_context: DaemonContext,
) -> None:
    daemon_context.telegram.get_updates = AsyncMock(return_value=[])
    offset = await _initial_offset(daemon_context, now_ts=NOW_TS)
    assert offset is None


async def test_initial_offset_getupdates_failure_falls_back_to_none(
    daemon_context: DaemonContext,
) -> None:
    daemon_context.telegram.get_updates = AsyncMock(side_effect=RuntimeError("net"))
    offset = await _initial_offset(daemon_context, now_ts=NOW_TS)
    assert offset is None


async def test_fresh_backlog_command_reaches_dispatch_not_dropped(
    daemon_context: DaemonContext,
) -> None:
    # End-to-end: compose the startup offset + first poll exactly as poll_commands
    # does; the fresh backlog command must reach dispatch (the bug dropped it).
    daemon_context.settings.telegram.chat_id = "5356256463"
    fresh = _update(41, "5356256463", "/status", date=int(NOW_TS) - 5)
    daemon_context.telegram.get_updates = AsyncMock(return_value=[fresh])
    daemon_context.telegram.send_message = AsyncMock()
    with patch(
        "optionsbot.daemon.telegram_poller.dispatch", new=AsyncMock(return_value=[])
    ) as disp:
        offset = await _initial_offset(daemon_context, now_ts=NOW_TS)
        await poll_once(daemon_context, offset)
    disp.assert_awaited_once()
    assert disp.await_args.args[1] == "/status"


async def test_poll_once_dispatches_authorized_and_advances_offset(
    daemon_context: DaemonContext,
) -> None:
    daemon_context.settings.telegram.chat_id = "5356256463"
    daemon_context.telegram.get_updates = AsyncMock(
        return_value=[_update(10, "5356256463", "/help")]
    )
    daemon_context.telegram.send_message = AsyncMock()
    new_offset = await poll_once(daemon_context, offset=0)
    assert new_offset == 11
    daemon_context.telegram.send_message.assert_awaited()  # a reply was sent


async def test_poll_once_ignores_foreign_chat(daemon_context: DaemonContext) -> None:
    daemon_context.settings.telegram.chat_id = "5356256463"
    daemon_context.telegram.get_updates = AsyncMock(
        return_value=[_update(10, "999", "/status")]
    )
    daemon_context.telegram.send_message = AsyncMock()
    new_offset = await poll_once(daemon_context, offset=0)
    assert new_offset == 11  # offset still advances
    daemon_context.telegram.send_message.assert_not_awaited()  # but no reply


async def test_poll_once_handler_error_is_swallowed_and_replied(
    daemon_context: DaemonContext,
) -> None:
    daemon_context.settings.telegram.chat_id = "5356256463"
    daemon_context.telegram.get_updates = AsyncMock(
        return_value=[_update(10, "5356256463", "/status")]
    )
    daemon_context.telegram.send_message = AsyncMock()
    with patch(
        "optionsbot.daemon.telegram_poller.dispatch",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        new_offset = await poll_once(daemon_context, offset=0)
    assert new_offset == 11
    sent = daemon_context.telegram.send_message.await_args.args[0]
    assert "failed" in sent.lower()


async def test_poll_once_bare_scan_skips_ack(daemon_context: DaemonContext) -> None:
    """Bare /scan (no symbol) must NOT get the 'scanning…' ack before the usage hint."""
    daemon_context.settings.telegram.chat_id = "5356256463"
    daemon_context.telegram.get_updates = AsyncMock(
        return_value=[_update(10, "5356256463", "/scan")]
    )
    daemon_context.telegram.send_message = AsyncMock()
    await poll_once(daemon_context, offset=0)
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert not any("scanning" in s for s in sent)  # no misleading ack for a bare /scan
    assert any("usage" in s.lower() for s in sent)  # only the usage hint
