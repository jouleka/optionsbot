"""Telegram command registry + dispatcher (IBK-102).

Handlers are pure-ish: they take a DaemonContext + parsed args and return a
list of CommandReply. No Telegram/poller wiring lives here, so the whole
surface is unit-testable with a mocked context. The poller sends the replies.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.market_hours import is_market_open
from optionsbot.storage.schema import alerts, scan_runs, watchlist


@dataclass(frozen=True, slots=True)
class CommandReply:
    text: str
    parse_mode: str | None = None  # None = plain text; "MarkdownV2" for picks


Handler = Callable[[DaemonContext, list[str]], Awaitable[list[CommandReply]]]

_HELP = (
    "optionsbot commands:\n"
    "/status — daemon health + last tick\n"
    "/last [N] — recent alerts (default 5)\n"
    "/scan SYMBOL — scan one symbol now\n"
    "/screen [N] — screener top N\n"
    "/pause — stop alerting\n"
    "/resume — resume alerting\n"
    "/watchlist list|add SYM|remove SYM\n"
    "/help — this message"
)


def _fmt_duration(delta: timedelta) -> str:  # timedelta -> compact "2h 5m"
    total = int(delta.total_seconds())
    h, rem = divmod(total, 3600)
    m = rem // 60
    return f"{h}h {m}m" if h else f"{m}m"


async def _cmd_help(context: DaemonContext, args: list[str]) -> list[CommandReply]:
    return [CommandReply(_HELP)]


async def _cmd_status(context: DaemonContext, args: list[str]) -> list[CommandReply]:
    now = datetime.now(UTC)
    with context.engine.connect() as conn:
        last = conn.execute(
            select(scan_runs).order_by(scan_runs.c.id.desc()).limit(1)
        ).first()
        wl = conn.execute(select(func.count()).select_from(watchlist)).scalar() or 0
    lines = [
        f"daemon up {_fmt_duration(now - context.started_at)}",
        f"IBKR: {'connected' if context.ibkr.is_connected else 'disconnected'}",
        f"market: {'open' if is_market_open(now) else 'closed'}",
        f"alerting: {'PAUSED' if context.alerting_paused else 'on'}",
        f"watchlist: {wl} symbol{'s' if wl != 1 else ''}",
    ]
    if last is not None:
        lines.append(
            f"last tick: {last.finished:%H:%M}Z scanned {last.tickers_scanned} "
            f"alerts {last.alerts_fired}"
        )
    else:
        lines.append("last tick: none yet")
    return [CommandReply("\n".join(lines))]


async def _cmd_last(context: DaemonContext, args: list[str]) -> list[CommandReply]:
    n = int(args[0]) if args and args[0].isdigit() else 5
    n = max(1, min(n, 20))
    with context.engine.connect() as conn:
        rows = conn.execute(
            select(alerts).order_by(alerts.c.id.desc()).limit(n)
        ).fetchall()
    if not rows:
        return [CommandReply("no alerts yet")]
    lines = [
        f"{r.symbol} {r.strategy} score {r.score:.0f} [{r.status}] {r.ts:%m-%d %H:%M}"
        for r in rows
    ]
    return [CommandReply("recent alerts:\n" + "\n".join(lines))]


async def _cmd_pause(context: DaemonContext, args: list[str]) -> list[CommandReply]:
    context.alerting_paused = True
    return [CommandReply("⏸ alerting paused (scans continue). /resume to re-enable.")]


async def _cmd_resume(context: DaemonContext, args: list[str]) -> list[CommandReply]:
    context.alerting_paused = False
    return [CommandReply("▶ alerting resumed.")]


_REGISTRY: dict[str, Handler] = {
    "help": _cmd_help,
    "status": _cmd_status,
    "last": _cmd_last,
    "pause": _cmd_pause,
    "resume": _cmd_resume,
}


async def dispatch(context: DaemonContext, text: str) -> list[CommandReply]:
    """Parse ``text`` and run the matching handler. Pure of any I/O beyond the
    handlers' own DB/IBKR reads. Unknown/blank input returns a help hint."""
    parts = text.split()
    if not parts or not parts[0].startswith("/"):
        return [CommandReply("send a command — try /help")]
    cmd = parts[0].lstrip("/").lower()
    handler = _REGISTRY.get(cmd)
    if handler is None:
        return [CommandReply(f"unknown command: /{cmd} — try /help")]
    return await handler(context, parts[1:])
