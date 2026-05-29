"""Tests for the Typer CLI scaffold."""

from typer.testing import CliRunner

from optionsbot.cli import app

runner = CliRunner()


def test_root_help_lists_expected_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("init", "status", "scan-once", "watch"):
        assert cmd in result.stdout


def test_no_login_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert "login" not in result.stdout.lower()


def test_watch_has_three_subcommands() -> None:
    result = runner.invoke(app, ["watch", "--help"])
    assert result.exit_code == 0
    for sub in ("add", "remove", "list"):
        assert sub in result.stdout
