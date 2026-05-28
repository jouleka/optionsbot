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


def test_scan_once_stub_exits_with_known_message() -> None:
    result = runner.invoke(app, ["scan-once"])
    assert result.exit_code != 0
    assert "IBK-7" in result.stdout or "IBK-7" in (result.stderr or "")


def test_watch_has_three_subcommands() -> None:
    result = runner.invoke(app, ["watch", "--help"])
    assert result.exit_code == 0
    for sub in ("add", "remove", "list"):
        assert sub in result.stdout


def test_watch_add_stub_exits_with_known_message() -> None:
    result = runner.invoke(app, ["watch", "add", "SPY"])
    assert result.exit_code != 0
    assert "IBK-51" in result.stdout or "IBK-51" in (result.stderr or "")
