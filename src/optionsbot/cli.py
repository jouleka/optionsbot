"""Optionsbot CLI. Fleshed out in IBK-14 (this plan, Task 5)."""
import typer

app = typer.Typer(help="Optionsbot: IBKR options analysis and alerts (paper trading).")


@app.callback()
def _root() -> None:
    """Root callback (required so `app` has a body even before subcommands are added)."""
    return
