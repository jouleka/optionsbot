"""Inbound Telegram command poller (IBK-102).

Long-polls getUpdates inside the daemon event loop, authorizes by chat_id, and
dispatches to the command registry. The loop never dies on transient errors and
exits cleanly on cancellation (daemon shutdown). Pre-startup backlog is processed
only when fresh (e.g. a command sent seconds before a restart); genuinely-stale
backlog is dropped so old commands never replay on startup.
"""

from __future__ import annotations

import asyncio
import logging
import time

from optionsbot.daemon.commands import CommandReply, dispatch
from optionsbot.daemon.context import DaemonContext

log = logging.getLogger(__name__)

_POLL_TIMEOUT = 30
_ERROR_BACKOFF = 5.0
# Process a command sent within this window before startup (e.g. one issued
# seconds before a restart); drop genuinely-stale backlog older than this.
_BACKLOG_MAX_AGE_S = 120
_ACK = {
    "scan": "⏳ scanning, one moment…",
    "screen": "⏳ screening the universe…",
    "execute": "⏳ checking gates and fresh quotes…",
}


async def _initial_offset(
    context: DaemonContext, now_ts: float | None = None
) -> int | None:
    """Starting offset for the command loop.

    Telegram retains commands sent while the daemon was down/restarting. We honor
    a command sent shortly before a restart (message newer than
    ``_BACKLOG_MAX_AGE_S``) but still DROP genuinely-stale backlog so an hours-old
    /execute can never fire on startup. Returns the offset to begin polling from:
    the first fresh update's id (process it + everything after), or last+1 when
    all backlog is stale.

    Replay safety: this can re-deliver a command that was processed just before
    the daemon died (Telegram confirms an offset only on the next getUpdates).
    That is harmless — the only POSITION-MUTATING commands are idempotent against
    replay: /execute refuses a pick that already has an order, /close refuses a
    position already closing. Other replays at worst re-run a read/scan or
    re-send a message the user already saw.
    """
    now_ts = now_ts if now_ts is not None else time.time()
    try:
        updates = await context.telegram.get_updates(offset=None, timeout=0)
    except Exception:  # noqa: BLE001 -- best-effort; fall back to default offset
        log.exception("getUpdates backlog scan failed; starting from default offset")
        return None
    if not updates:
        return None
    for update in updates:
        message = update.get("message") or {}
        date = message.get("date")
        # A future-dated message (clock skew) gives a negative age and counts as
        # fresh — the safe direction (process a recent command, don't drop it).
        if date is not None and (now_ts - float(date)) <= _BACKLOG_MAX_AGE_S:
            return int(update["update_id"])  # process from the first fresh command
    return int(updates[-1]["update_id"]) + 1  # all stale -> drop the backlog


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
    offset = await _initial_offset(context)
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
