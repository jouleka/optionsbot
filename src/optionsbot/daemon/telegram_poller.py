"""Inbound Telegram command poller (IBK-102).

Long-polls getUpdates inside the daemon event loop, authorizes by chat_id, and
dispatches to the command registry. The loop never dies on transient errors and
exits cleanly on cancellation (daemon shutdown). Backlog from while the daemon
was down is dropped on startup.
"""

from __future__ import annotations

import asyncio
import logging

from optionsbot.daemon.commands import CommandReply, dispatch
from optionsbot.daemon.context import DaemonContext

log = logging.getLogger(__name__)

_POLL_TIMEOUT = 30
_ERROR_BACKOFF = 5.0
_ACK = {
    "scan": "⏳ scanning, one moment…",
    "screen": "⏳ screening the universe…",
    "execute": "⏳ checking gates and fresh quotes…",
}


async def _drop_backlog(context: DaemonContext) -> int | None:
    """Return the starting offset, skipping any updates queued before startup."""
    try:
        updates = await context.telegram.get_updates(offset=-1, timeout=0)
    except Exception:  # noqa: BLE001 -- best-effort; fall back to default offset
        log.exception("getUpdates backlog drop failed; starting from default offset")
        return None
    if updates:
        return int(updates[-1]["update_id"]) + 1
    return None


async def poll_once(context: DaemonContext, offset: int | None) -> int | None:
    """One getUpdates round: dispatch authorized commands, return the next offset.

    Raised separately from the loop so it is directly unit-testable.
    """
    updates = await context.telegram.get_updates(offset=offset, timeout=_POLL_TIMEOUT)
    chat_id = str(context.settings.telegram.chat_id)
    for update in updates:
        offset = int(update["update_id"]) + 1
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        if str(chat.get("id")) != chat_id:
            log.debug("ignoring update from unauthorized chat %s", chat.get("id"))
            continue
        text = (message.get("text") or "").strip()
        if not text:
            continue
        parts = text.split()
        cmd = parts[0].lstrip("/").lower() if text.startswith("/") else ""
        # Only ack a slow command that will actually do work (e.g. "/scan SPY",
        # not bare "/scan", which just returns a usage hint).
        if cmd in _ACK and len(parts) > 1:
            await context.telegram.send_message(_ACK[cmd], parse_mode=None)
        try:
            replies = await dispatch(context, text)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- a bad command must not kill the loop
            log.exception("command failed: %s", text)
            replies = [CommandReply(f"⚠️ command failed: {type(e).__name__}")]
        for reply in replies:
            await context.telegram.send_message(reply.text, parse_mode=reply.parse_mode)
    return offset


async def poll_commands(context: DaemonContext) -> None:
    """Run the inbound command loop until cancelled."""
    offset = await _drop_backlog(context)
    log.info("telegram command poller started")
    while True:
        try:
            offset = await poll_once(context, offset)
        except asyncio.CancelledError:
            log.info("telegram command poller stopping")
            raise
        except Exception:  # noqa: BLE001 -- transient getUpdates/network errors
            log.exception("poll_once failed; backing off %.0fs", _ERROR_BACKOFF)
            await asyncio.sleep(_ERROR_BACKOFF)
