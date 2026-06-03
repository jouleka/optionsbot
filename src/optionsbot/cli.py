"""Optionsbot CLI scaffold. Stubs only -- real implementations land in later epics."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from optionsbot.config import Settings

app = typer.Typer(
    help="Optionsbot: IBKR options analysis and alerts (paper trading).",
    no_args_is_help=True,
)
watch_app = typer.Typer(
    help="Manage the watchlist (configurable via chat in IBK-6 too).",
    no_args_is_help=True,
)
app.add_typer(watch_app, name="watch")


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_TOML = """\
# optionsbot configuration. Env vars (prefix OPTIONSBOT_, nested via __)
# override these values at runtime.

# log_level lives at the TOP because Settings.log_level is a top-level field.
# Putting it under any [section] header would bind it to that section --
# pydantic would silently drop the stray key and the user's edit would
# appear to have no effect.
log_level = "INFO"

[ibkr]
host = "127.0.0.1"
port = 4002         # 4002 = paper IB Gateway; 4001 = live (do not use)
paper = true
# client_id_mcp = 1
# client_id_daemon = 2

[scan]
interval_minutes = 15
score_threshold = 70
alert_cooldown_hours = 4
alert_rescore_delta = 10

[telegram]
# bot_token = "123456:ABC-..."
# chat_id = "123456789"

[storage]
# db_path = "~/.local/share/optionsbot/optionsbot.db"
"""

_TELEGRAM_SECTION_RE = re.compile(
    r"^\[telegram\][^\[]*",
    re.MULTILINE,
)


def _ensure_config_dir(override: Path | None) -> Path:
    """Create the config dir if missing. Return its absolute Path."""
    cfg_dir = (
        override if override is not None else Path.home() / ".config" / "optionsbot"
    ).expanduser()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"config dir: {cfg_dir}")
    return cfg_dir


def _write_default_config(cfg_dir: Path, non_interactive: bool) -> Path:
    """Write the default config.toml if missing. Return its path.

    If the file already exists, prompt before overwriting (or skip in
    non-interactive mode).
    """
    cfg_path = cfg_dir / "config.toml"
    if cfg_path.exists():
        if non_interactive:
            typer.echo(f"config exists, leaving in place: {cfg_path}")
            return cfg_path
        if not typer.confirm(
            f"config.toml exists at {cfg_path}. Overwrite with defaults?",
            default=False,
        ):
            typer.echo("keeping existing config")
            return cfg_path
    cfg_path.write_text(_DEFAULT_CONFIG_TOML)
    typer.echo(f"wrote default config: {cfg_path}")
    return cfg_path


def _run_migrations() -> None:
    """Run alembic upgrade head against the configured DB path."""
    from alembic.config import Config

    from alembic import command
    from optionsbot.config import get_settings, load_settings

    # Use load_settings so a freshly-written config.toml is read.
    get_settings.cache_clear()
    settings = load_settings()
    db_path = settings.storage.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[2]
    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")
    typer.echo(f"db migrated: {db_path}")


def _persist_telegram_creds(cfg_path: Path, token: str, chat_id: str) -> None:
    """Replace (or append) the [telegram] section in config.toml with the
    provided credentials. Preserves the rest of the file."""
    body = cfg_path.read_text()
    new_section = (
        f"[telegram]\n"
        f'bot_token = "{token}"\n'
        f'chat_id = "{chat_id}"\n'
    )
    if _TELEGRAM_SECTION_RE.search(body):
        body = _TELEGRAM_SECTION_RE.sub(new_section, body, count=1)
    else:
        if not body.endswith("\n"):
            body += "\n"
        body += "\n" + new_section
    cfg_path.write_text(body)
    typer.echo(f"  saved Telegram credentials to {cfg_path}")


async def _send_telegram_test(token: str, chat_id: str) -> int | None:
    """Try to send 'optionsbot init test message'. Return msg_id or None."""
    from optionsbot.daemon.telegram_client import TelegramClient

    client = TelegramClient(token, chat_id)
    try:
        msg_id = await client.send_message("optionsbot init test message")
        return msg_id
    except Exception as e:  # noqa: BLE001 -- surface a friendly error
        msg = str(e)
        hint = ""
        if "401" in msg or "Unauthorized" in msg:
            hint = " (check bot_token)"
        elif "400" in msg or "chat not found" in msg.lower():
            hint = " (start a chat with your bot first, then re-run)"
        typer.secho(
            f"  ✗ Telegram send failed: {type(e).__name__}: {msg}{hint}",
            fg=typer.colors.RED,
        )
        return None
    finally:
        await client.aclose()


async def _configure_telegram(
    cfg_path: Path, non_interactive: bool, skip_test: bool
) -> None:
    """Prompt for Telegram credentials + optionally test send.

    In non-interactive mode, uses whatever's already in config.toml /
    env vars (no prompts). If credentials are blank after that, prints
    a notice and returns. async so we can `await _send_telegram_test`
    directly without nesting asyncio.run inside the outer asyncio.run.
    """
    from optionsbot.config import get_settings, load_settings

    get_settings.cache_clear()
    settings = load_settings()
    token = settings.telegram.bot_token
    chat_id = settings.telegram.chat_id

    if not non_interactive and not (token and chat_id):
        typer.echo("\nTelegram setup (skip with --skip-telegram):")
        token_in = typer.prompt(
            "  bot_token (leave blank to skip)",
            default="",
            show_default=False,
        ).strip()
        chat_in = typer.prompt(
            "  chat_id (leave blank to skip)",
            default="",
            show_default=False,
        ).strip()
        if token_in and chat_in:
            _persist_telegram_creds(cfg_path, token_in, chat_in)
            token = token_in
            chat_id = chat_in
            # Re-load so subsequent steps see the new values.
            get_settings.cache_clear()

    if not (token and chat_id):
        typer.echo(
            "Telegram not configured -- daemon alerts will be disabled"
            " until you set bot_token + chat_id."
        )
        return

    if skip_test:
        typer.echo("Telegram configured; skipping test send (--skip-test).")
        return

    typer.echo("Sending Telegram test message...")
    msg_id = await _send_telegram_test(token, chat_id)
    if msg_id is not None:
        typer.secho(f"  ✓ message sent (id={msg_id})", fg=typer.colors.GREEN)


def _final_summary(cfg_dir: Path) -> str:
    return (
        f"setup complete.\n\n"
        f"next steps:\n"
        f"  1. Review {cfg_dir / 'config.toml'} -- adjust ibkr.port if needed (4002 = paper).\n"
        f"  2. Start IB Gateway on the configured port (paper account).\n"
        f"  3. Run `optionsbot status` to verify connectivity.\n"
        f"  4. Run `optionsbot-daemon` to start the scheduler.\n"
    )


async def _run_init(
    non_interactive: bool,
    skip_telegram: bool,
    skip_test: bool,
    config_dir: Path | None,
) -> None:
    typer.secho("optionsbot setup", fg=typer.colors.GREEN, bold=True)
    cfg_dir = _ensure_config_dir(config_dir)
    cfg_path = _write_default_config(cfg_dir, non_interactive)
    _run_migrations()
    if not skip_telegram:
        await _configure_telegram(cfg_path, non_interactive, skip_test)
    typer.echo("\n" + _final_summary(cfg_dir))


@app.command()
def init(
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Skip prompts; use existing config + env vars only.",
    ),
    skip_telegram: bool = typer.Option(
        False,
        "--skip-telegram",
        help="Don't prompt for Telegram credentials and don't try to send a test message.",
    ),
    skip_test: bool = typer.Option(
        False,
        "--skip-test",
        help="Skip the Telegram send test even if credentials are configured.",
    ),
    config_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--config-dir",
        help="Override the default ~/.config/optionsbot/ location (mainly for tests).",
    ),
) -> None:
    """Interactive setup: config dir, DB migrations, Telegram credentials."""
    asyncio.run(_run_init(non_interactive, skip_telegram, skip_test, config_dir))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@dataclass
class _CheckResult:
    name: str
    state: str  # "ok" | "warn" | "fail"
    detail: str
    # True when this subsystem is configured (or always-on like db/ibkr) and
    # should gate the process exit code. False when the user has skipped it
    # or it's not configured -- exit code unaffected by state in that case.
    is_critical: bool = True


@app.command()
def status(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of pretty text."
    ),
    no_telegram: bool = typer.Option(
        False, "--no-telegram", help="Skip the Telegram getMe check."
    ),
) -> None:
    """Health check: DB, IB Gateway, daemon, Telegram bot reachability."""
    code = asyncio.run(_run_status(json_output=json_output, no_telegram=no_telegram))
    raise typer.Exit(code=code)


async def _run_status(*, json_output: bool, no_telegram: bool) -> int:
    from optionsbot.config import get_settings, load_settings

    get_settings.cache_clear()
    settings = load_settings()

    db_check = _check_db(settings)
    ibkr_check = _check_ibkr_socket(settings)
    last_scan = _check_last_scan(settings)
    last_alert = _check_last_alert(settings)
    if no_telegram:
        tg_check = _CheckResult(
            "telegram", "warn", "skipped (--no-telegram)", is_critical=False
        )
    else:
        tg_check = await _check_telegram(settings)

    # last scan / last alert are informational; exit code shouldn't depend on them.
    last_scan.is_critical = False
    last_alert.is_critical = False

    results = [db_check, ibkr_check, last_scan, last_alert, tg_check]

    if json_output:
        typer.echo(json.dumps([asdict(r) for r in results], indent=2))
    else:
        for r in results:
            sigil = {"ok": "✓", "warn": "⚠", "fail": "✗"}[r.state]
            color = {
                "ok": typer.colors.GREEN,
                "warn": typer.colors.YELLOW,
                "fail": typer.colors.RED,
            }[r.state]
            typer.secho(f"{sigil} {r.name:14s} {r.detail}", fg=color)

    # Exit non-zero iff any CRITICAL subsystem failed. Uses an explicit
    # is_critical flag rather than a brittle string-match on detail text.
    if any(c.is_critical and c.state == "fail" for c in results):
        return 1
    return 0


def _check_db(settings) -> _CheckResult:  # type: ignore[no-untyped-def]
    from sqlalchemy import func, select

    from optionsbot.storage.db import create_engine_for_path
    from optionsbot.storage.schema import watchlist

    db_path = settings.storage.db_path
    if not db_path.exists():
        return _CheckResult(
            "db", "fail", f"{db_path} does not exist (run `optionsbot init`)"
        )
    try:
        engine = create_engine_for_path(db_path)
        with engine.connect() as conn:
            count = (
                conn.execute(select(func.count()).select_from(watchlist)).scalar() or 0
            )
        return _CheckResult("db", "ok", f"{db_path} ({count} watchlist entries)")
    except Exception as e:  # noqa: BLE001
        return _CheckResult("db", "fail", f"{db_path}: {type(e).__name__}: {e}")


def _check_ibkr_socket(settings) -> _CheckResult:  # type: ignore[no-untyped-def]
    import socket

    host = settings.ibkr.host
    port = settings.ibkr.port
    label = "paper" if settings.ibkr.paper else "live"
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return _CheckResult("ibkr", "ok", f"{host}:{port} ({label})")
    except (OSError, TimeoutError) as e:
        return _CheckResult(
            "ibkr",
            "fail",
            f"{host}:{port} ({label}): {type(e).__name__}: {e}",
        )


def _check_last_scan(settings) -> _CheckResult:  # type: ignore[no-untyped-def]
    from sqlalchemy import func, select

    from optionsbot.storage.db import create_engine_for_path
    from optionsbot.storage.schema import scan_runs

    if not settings.storage.db_path.exists():
        return _CheckResult("last scan", "warn", "db missing")
    engine = create_engine_for_path(settings.storage.db_path)
    try:
        with engine.connect() as conn:
            last = conn.execute(select(func.max(scan_runs.c.started))).scalar()
    except Exception as e:  # noqa: BLE001
        return _CheckResult("last scan", "warn", f"query failed: {e}")
    if last is None:
        return _CheckResult("last scan", "warn", "never")
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    age = datetime.now(UTC) - last
    minutes = age.total_seconds() / 60
    threshold = 2 * settings.scan.interval_minutes
    state = "ok" if minutes <= threshold else "warn"
    return _CheckResult(
        "last scan",
        state,
        f"{last.isoformat()} ({minutes:.0f}m ago, threshold {threshold}m)",
    )


def _check_last_alert(settings) -> _CheckResult:  # type: ignore[no-untyped-def]
    from sqlalchemy import func, select

    from optionsbot.storage.db import create_engine_for_path
    from optionsbot.storage.schema import alerts

    if not settings.storage.db_path.exists():
        return _CheckResult("last alert", "warn", "db missing")
    engine = create_engine_for_path(settings.storage.db_path)
    try:
        with engine.connect() as conn:
            last = conn.execute(
                select(func.max(alerts.c.ts)).where(alerts.c.status == "sent")
            ).scalar()
    except Exception as e:  # noqa: BLE001
        return _CheckResult("last alert", "warn", f"query failed: {e}")
    if last is None:
        return _CheckResult("last alert", "warn", "none sent yet")
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    age = datetime.now(UTC) - last
    return _CheckResult(
        "last alert",
        "ok",
        f"{last.isoformat()} ({age.total_seconds() / 60:.0f}m ago)",
    )


async def _check_telegram(settings) -> _CheckResult:  # type: ignore[no-untyped-def]
    import httpx

    token = settings.telegram.bot_token
    chat_id = settings.telegram.chat_id
    if not (token and chat_id):
        # Not configured -> warn but don't gate the exit code (is_critical=False).
        return _CheckResult("telegram", "warn", "not configured", is_critical=False)
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            username = data.get("result", {}).get("username", "?")
            return _CheckResult("telegram", "ok", f"@{username}")
    except Exception as e:  # noqa: BLE001
        return _CheckResult("telegram", "fail", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# watch + scan-once
# ---------------------------------------------------------------------------


def _load_settings_and_engine() -> tuple[Settings, Engine]:
    """Fresh Settings + an Engine for the configured DB. Exits 1 if the DB is missing."""
    from optionsbot.config import get_settings, load_settings
    from optionsbot.storage.db import create_engine_for_path

    get_settings.cache_clear()
    settings = load_settings()
    if not settings.storage.db_path.exists():
        typer.secho(
            "db not found -- run `optionsbot init` first.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)
    return settings, create_engine_for_path(settings.storage.db_path)


@watch_app.command("list")
def watch_list() -> None:
    """List all tickers in the watchlist."""
    from sqlalchemy import select

    from optionsbot.storage.schema import watchlist

    _settings, engine = _load_settings_and_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                watchlist.c.symbol,
                watchlist.c.view_override_dir,
                watchlist.c.view_override_iv,
                watchlist.c.notes,
            ).order_by(watchlist.c.symbol)
        ).fetchall()
    if not rows:
        typer.echo("watchlist is empty -- add one with `optionsbot watch add SYMBOL`")
        return
    for r in rows:
        bits: list[str] = []
        if r.view_override_dir or r.view_override_iv:
            bits.append(f"view={r.view_override_dir or '-'}/{r.view_override_iv or '-'}")
        if r.notes:
            bits.append(f"note={r.notes}")
        suffix = ("  " + " ".join(bits)) if bits else ""
        typer.echo(f"{r.symbol}{suffix}")


@watch_app.command("remove")
def watch_remove(symbol: str) -> None:
    """Remove a ticker from the watchlist. Snapshot history is preserved."""
    from sqlalchemy import delete

    from optionsbot.storage.schema import watchlist

    symbol = symbol.upper().strip()
    _settings, engine = _load_settings_and_engine()
    with engine.begin() as conn:
        result = conn.execute(delete(watchlist).where(watchlist.c.symbol == symbol))
    if result.rowcount > 0:
        typer.secho(f"removed {symbol} from the watchlist", fg=typer.colors.GREEN)
    else:
        typer.echo(f"{symbol} is not in the watchlist")


@watch_app.command("add")
def watch_add(
    symbol: str,
    notes: str | None = typer.Option(
        None, "--notes", help="Optional note stored with the entry."
    ),
) -> None:
    """Add a ticker to the watchlist (validated against IBKR first)."""
    raise typer.Exit(code=asyncio.run(_run_watch_add(symbol, notes)))


async def _run_watch_add(symbol: str, notes: str | None) -> int:
    from sqlalchemy import insert
    from sqlalchemy.exc import IntegrityError

    from optionsbot.ibkr import IBKRClient
    from optionsbot.ibkr.contracts import ContractResolver
    from optionsbot.storage.schema import watchlist

    symbol = symbol.upper().strip()
    if not symbol:
        typer.secho("symbol is empty", fg=typer.colors.RED, err=True)
        return 2
    settings, engine = _load_settings_and_engine()

    # Validate the symbol against IBKR before persisting (mirrors the MCP tool).
    client = IBKRClient(role="cli", settings=settings)
    try:
        await client.connect()
        await ContractResolver(client).stock(symbol)
    except ConnectionError as e:
        typer.secho(f"IBKR unavailable: {e}", fg=typer.colors.RED, err=True)
        return 1
    except ValueError:
        typer.secho(
            f"unknown symbol: {symbol} (IBKR could not qualify it)",
            fg=typer.colors.RED,
            err=True,
        )
        return 1
    finally:
        await client.disconnect()

    with engine.begin() as conn:
        try:
            conn.execute(
                insert(watchlist).values(
                    symbol=symbol, notes=notes, added_at=datetime.now(UTC)
                )
            )
        except IntegrityError:
            typer.echo(f"{symbol} is already in the watchlist")
            return 0
    typer.secho(f"added {symbol} to the watchlist", fg=typer.colors.GREEN)
    return 0


@app.command("scan-once")
def scan_once() -> None:
    """Run a single scan over the watchlist and exit (no daemon, no alerts)."""
    raise typer.Exit(code=asyncio.run(_run_scan_once()))


async def _run_scan_once() -> int:
    from sqlalchemy import insert, select

    from optionsbot.ibkr import IBKRClient
    from optionsbot.ibkr.contracts import ContractResolver
    from optionsbot.scan import scan_symbol
    from optionsbot.scoring import DEFAULT_THRESHOLD, DEFAULT_TOP_K, top_k
    from optionsbot.storage.schema import scan_runs, watchlist

    settings, engine = _load_settings_and_engine()
    with engine.connect() as conn:
        symbols = [
            r.symbol
            for r in conn.execute(
                select(watchlist.c.symbol).order_by(watchlist.c.symbol)
            ).fetchall()
        ]
    if not symbols:
        typer.echo("watchlist is empty -- add one with `optionsbot watch add SYMBOL`")
        return 0

    client = IBKRClient(role="cli", settings=settings)
    started = datetime.now(UTC)
    scanned = 0
    errors: list[str] = []
    try:
        await client.connect()
        resolver = ContractResolver(client)
        for sym in symbols:
            try:
                result = await scan_symbol(sym, client, engine, settings, resolver=resolver)
            except Exception as e:  # noqa: BLE001 -- one symbol must not abort the sweep
                typer.secho(
                    f"{sym}: {type(e).__name__}: {e}", fg=typer.colors.RED, err=True
                )
                errors.append(f"{sym}: {type(e).__name__}: {e}")
                continue
            scanned += 1
            selected = top_k(result.scored, k=DEFAULT_TOP_K, threshold=DEFAULT_THRESHOLD)
            if selected:
                typer.secho(f"{sym}: {len(result.scored)} scored", fg=typer.colors.GREEN)
                for s in selected:
                    typer.echo(f"   {s.strategy_name}: {s.score:.0f}")
            else:
                typer.echo(
                    f"{sym}: {len(result.scored)} scored "
                    f"(none >= threshold {DEFAULT_THRESHOLD})"
                )
    finally:
        await client.disconnect()

    finished = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            insert(scan_runs).values(
                started=started,
                finished=finished,
                tickers_scanned=scanned,
                alerts_fired=0,
                errors_json=errors or None,
            )
        )
    typer.echo(f"scanned {scanned}/{len(symbols)} symbol(s)")
    return 0


@app.command()
def screen(
    top: int | None = typer.Option(
        None, "--top", min=1, help="How many candidates to show (default: config screener.top_n)."
    ),
) -> None:
    """Rank the configured universe by realized-vol (HV) rank + liquidity (no chains)."""
    raise typer.Exit(code=asyncio.run(_run_screen(top)))


async def _run_screen(top: int | None) -> int:
    from optionsbot.config import get_settings, load_settings
    from optionsbot.ibkr import IBKRClient
    from optionsbot.ibkr.history import HistoryClient
    from optionsbot.screener.screen import screen_universe
    from optionsbot.screener.universe import DEFAULT_UNIVERSE

    get_settings.cache_clear()
    settings = load_settings()
    universe = settings.screener.universe or list(DEFAULT_UNIVERSE)
    top_n = top if top is not None else settings.screener.top_n

    client = IBKRClient(role="cli", settings=settings)
    try:
        await client.connect()
        history = HistoryClient(client)
        candidates = await screen_universe(
            history, universe, settings.screener.min_dollar_volume
        )
    finally:
        await client.disconnect()

    if not candidates:
        typer.echo("no candidates passed the liquidity gate")
        return 0
    typer.secho(
        f"top {min(top_n, len(candidates))} of {len(candidates)} candidates:",
        fg=typer.colors.GREEN,
    )
    for c in candidates[:top_n]:
        typer.echo(f"  {c.symbol:6} hv_rank={c.hv_rank:.2f}  $vol={c.dollar_volume:,.0f}")
    return 0


validate_app = typer.Typer(help="Validate the bot's predictions (backtest, outcomes).")
app.add_typer(validate_app, name="validate")


@validate_app.command("backtest")
def validate_backtest(
    years: int = typer.Option(3, "--years", min=1, help="Years of underlying history."),
) -> None:
    """Backtest model prob_profit vs historically-realized win-rate over recorded picks."""
    raise typer.Exit(code=asyncio.run(_run_validate_backtest(years)))


async def _run_validate_backtest(years: int) -> int:
    from optionsbot.config import get_settings, load_settings
    from optionsbot.ibkr import IBKRClient
    from optionsbot.ibkr.history import HistoryClient
    from optionsbot.storage.db import create_engine_for_path
    from optionsbot.validation.backtest import load_pick_records, run_backtest

    get_settings.cache_clear()
    settings = load_settings()
    engine = create_engine_for_path(settings.storage.db_path)
    picks = load_pick_records(engine)
    if not picks:
        typer.echo("no terminal-modelable picks recorded yet")
        return 0

    client = IBKRClient(role="cli", settings=settings)
    # IB rejects day-bar durations beyond ~1 year ("756 D" -> 0 bars), so drive
    # the request in years. Under-request rows slightly (250/yr vs ~252 actual)
    # so the parquet cache satisfies on re-run instead of always re-fetching.
    duration = f"{years} Y"
    days = years * 250
    try:
        await client.connect()
        history = HistoryClient(client)

        async def fetch_closes(symbol: str) -> list[float]:
            df = await history.get_history(symbol, days=days, duration_str=duration)
            return [float(c) for c in df["close"].tolist()]

        report = await run_backtest(picks, fetch_closes)
    finally:
        await client.disconnect()

    typer.secho(
        f"Calibration over {report.overall_count} picks "
        f"(pred {report.overall_mean_pred:.2f} | dedrift {report.overall_mean_dedrift:.2f} | "
        f"raw {report.overall_mean_raw:.2f}):",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"  {'pred_PoP':>10} {'dedrift':>8} {'raw_win':>8} {'n_picks':>8}")
    for b in report.buckets:
        if b.count == 0:
            continue
        typer.echo(
            f"  [{b.lo:.1f},{b.hi:.1f}) {b.mean_pred:>9.2f} {b.mean_dedrift:>8.2f} "
            f"{b.mean_raw:>8.2f} {b.count:>8}"
        )
    typer.echo("by strategy:")
    for name, b in report.by_strategy.items():
        typer.echo(
            f"  {name:24} pred={b.mean_pred:.2f} dedrift={b.mean_dedrift:.2f} "
            f"raw={b.mean_raw:.2f} n={b.count}"
        )
    typer.echo(
        "note: judge calibration on pred-vs-dedrift (dedrift renormalizes realized "
        "returns to zero drift, matching the model); raw includes drift and is "
        "regime-specific (raw>dedrift = the bull/bear tailwind the model ignores). "
        "Win-rates use OVERLAPPING windows over a single ~Ny path, so effective "
        "independent samples are far fewer than n_picks -- read as directional, "
        "not a confidence interval."
    )
    return 0


if __name__ == "__main__":
    sys.exit(app())
