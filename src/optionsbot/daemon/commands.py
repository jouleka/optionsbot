"""Telegram command registry + dispatcher (IBK-102).

Handlers are pure-ish: they take a DaemonContext + parsed args and return a
list of CommandReply. No Telegram/poller wiring lives here, so the whole
surface is unit-testable with a mocked context. The poller sends the replies.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, insert, select

from optionsbot.alerts.formatter import (
    format_alert_markdown,
    format_positions_text,
    no_edge_note,
)
from optionsbot.analysis.positions import assemble_open_book
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.market_hours import is_market_open
from optionsbot.ibkr.history import HistoryClient
from optionsbot.ibkr.market_data import MarketDataClient
from optionsbot.ibkr.positions import PositionsClient
from optionsbot.scan import scan_symbol
from optionsbot.scoring.composite import edge_sort_key, has_positive_edge
from optionsbot.screener.screen import screen_universe
from optionsbot.screener.universe import DEFAULT_UNIVERSE
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
    "/positions — your open book (live P&L, DTE, Greeks)\n"
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


async def _cmd_scan(context: DaemonContext, args: list[str]) -> list[CommandReply]:
    if not args:
        return [CommandReply("usage: /scan SYMBOL")]
    symbol = args[0].upper()
    async with context.ibkr_lock:
        result = await scan_symbol(
            symbol, context.ibkr, context.engine, context.settings,
            resolver=context.resolver,
        )
    ranked = sorted(result.scored, key=lambda s: edge_sort_key(s.suggestion), reverse=True)
    top = ranked[:3]
    if not top:
        return [CommandReply(f"{symbol}: no qualifying strategies right now")]
    replies = [
        CommandReply(
            format_alert_markdown(result.symbol, result.view, s, result.snapshot_ts),
            parse_mode="MarkdownV2",
        )
        for s in top
    ]
    if not any(has_positive_edge(s.suggestion) for s in top):
        replies.insert(0, CommandReply(no_edge_note(result.symbol)))
    return replies


async def _cmd_screen(context: DaemonContext, args: list[str]) -> list[CommandReply]:
    n = int(args[0]) if args and args[0].isdigit() else context.settings.screener.top_n
    n = max(1, min(n, 30))
    history = HistoryClient(context.ibkr, context.resolver)
    universe = context.settings.screener.universe or DEFAULT_UNIVERSE
    async with context.ibkr_lock:
        cands = await screen_universe(
            history, universe, context.settings.screener.min_dollar_volume
        )
    if not cands:
        return [CommandReply("screen: no candidates passed the liquidity gate")]
    lines = ["top screened:"]
    lines += [
        f"{c.symbol}: hv_rank {c.hv_rank:.2f}, $vol {c.dollar_volume / 1e6:.0f}M"
        for c in cands[:n]
    ]
    return [CommandReply("\n".join(lines))]


async def _cmd_watchlist(context: DaemonContext, args: list[str]) -> list[CommandReply]:
    sub = args[0].lower() if args else "list"
    if sub == "list":
        with context.engine.connect() as conn:
            rows = conn.execute(
                select(watchlist.c.symbol).order_by(watchlist.c.symbol)
            ).fetchall()
        if not rows:
            return [CommandReply("watchlist is empty — /watchlist add SYM")]
        return [CommandReply("watchlist:\n" + "\n".join(r.symbol for r in rows))]
    if sub == "add":
        if len(args) < 2:
            return [CommandReply("usage: /watchlist add SYMBOL")]
        symbol = args[1].upper()
        try:
            async with context.ibkr_lock:
                await context.resolver.stock(symbol)  # validate; raises if unknown
        except Exception:  # noqa: BLE001
            return [CommandReply(f"could not validate {symbol} against IBKR")]
        with context.engine.begin() as conn:
            exists = conn.execute(
                select(watchlist.c.symbol).where(watchlist.c.symbol == symbol)
            ).first()
            if exists:
                return [CommandReply(f"{symbol} is already in the watchlist")]
            conn.execute(insert(watchlist).values(symbol=symbol, added_at=datetime.now(UTC)))
        return [CommandReply(f"added {symbol} to the watchlist")]
    if sub == "remove":
        if len(args) < 2:
            return [CommandReply("usage: /watchlist remove SYMBOL")]
        symbol = args[1].upper()
        with context.engine.begin() as conn:
            res = conn.execute(delete(watchlist).where(watchlist.c.symbol == symbol))
        if res.rowcount:
            return [CommandReply(f"removed {symbol} from the watchlist")]
        return [CommandReply(f"{symbol} is not in the watchlist")]
    return [CommandReply("usage: /watchlist list|add SYM|remove SYM")]


async def _cmd_positions(context: DaemonContext, args: list[str]) -> list[CommandReply]:
    pos_client = PositionsClient(context.ibkr)
    md_client = MarketDataClient(context.ibkr, context.resolver)
    try:
        async with context.ibkr_lock:
            view = await assemble_open_book(pos_client, md_client, datetime.now(UTC))
    except Exception:  # noqa: BLE001 -- surface a plain failure, never crash the poller
        return [CommandReply("couldn't reach IBKR for positions")]
    return [CommandReply(format_positions_text(view))]


_REGISTRY: dict[str, Handler] = {
    "help": _cmd_help,
    "status": _cmd_status,
    "last": _cmd_last,
    "pause": _cmd_pause,
    "resume": _cmd_resume,
    "scan": _cmd_scan,
    "screen": _cmd_screen,
    "watchlist": _cmd_watchlist,
    "positions": _cmd_positions,
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
