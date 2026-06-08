"""Tests for the `migrate` CLI command (IBK-113)."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from optionsbot.cli import app


def test_migrate_command_runs_migrations() -> None:
    with patch("optionsbot.cli._run_migrations") as m:
        result = CliRunner().invoke(app, ["migrate"])
    assert result.exit_code == 0
    m.assert_called_once()
