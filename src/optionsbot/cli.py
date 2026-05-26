"""Optionsbot CLI scaffold. Stubs only -- real implementations land in later epics."""

from __future__ import annotations

import sys

import typer

app = typer.Typer(
    help="Optionsbot: IBKR options analysis and alerts (paper trading).",
    no_args_is_help=True,
)
watch_app = typer.Typer(
    help="Manage the watchlist (configurable via chat in IBK-6 too).",
    no_args_is_help=True,
)
app.add_typer(watch_app, name="watch")


def _stub(name: str, epic: str) -> None:
    typer.echo(
        f"`optionsbot {name}` is not yet implemented. See {epic}.",
        err=True,
    )
    raise typer.Exit(code=2)


@app.command()
def init() -> None:
    """Interactive setup: config dir, DB migrations, Telegram credentials."""
    _stub("init", "IBK-73")


@app.command()
def status() -> None:
    """Health check: DB, IB Gateway, daemon, Telegram bot reachability."""
    _stub("status", "IBK-74")


@app.command("scan-once")
def scan_once() -> None:
    """Run a single scan over the watchlist and exit (no daemon)."""
    _stub("scan-once", "IBK-7 (Daemon)")


@watch_app.command("add")
def watch_add(symbol: str) -> None:
    """Add a ticker to the watchlist."""
    _stub(f"watch add {symbol}", "IBK-51")


@watch_app.command("remove")
def watch_remove(symbol: str) -> None:
    """Remove a ticker from the watchlist."""
    _stub(f"watch remove {symbol}", "IBK-52")


@watch_app.command("list")
def watch_list() -> None:
    """List all tickers in the watchlist."""
    _stub("watch list", "IBK-53")


if __name__ == "__main__":
    sys.exit(app())
