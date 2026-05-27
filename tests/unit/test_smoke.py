"""Smoke tests that the package and stub entry points are importable."""

import optionsbot
from optionsbot.cli import app as cli_app
from optionsbot.daemon import __main__ as daemon_main
from optionsbot.mcp_server import __main__ as mcp_main


def test_package_has_version() -> None:
    assert optionsbot.__version__ == "0.0.1"


def test_cli_app_is_typer() -> None:
    import typer
    assert isinstance(cli_app, typer.Typer)


def test_mcp_main_is_callable() -> None:
    """mcp_main.main is a callable (real implementation replaces the old stub)."""
    assert callable(mcp_main.main)


def test_daemon_main_returns_nonzero_for_now() -> None:
    assert daemon_main.main() == 1
