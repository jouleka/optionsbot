from __future__ import annotations

from unittest.mock import AsyncMock, patch

from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.telegram_poller import _drop_backlog, poll_once


def _update(uid: int, chat_id: str, text: str) -> dict:
    return {"update_id": uid, "message": {"chat": {"id": int(chat_id)}, "text": text}}


async def test_drop_backlog_advances_past_last(daemon_context: DaemonContext) -> None:
    updates = [_update(41, "5356256463", "/status")]
    daemon_context.telegram.get_updates = AsyncMock(return_value=updates)
    daemon_context.settings.telegram.chat_id = "5356256463"
    offset = await _drop_backlog(daemon_context)
    assert offset == 42


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
