"""Tests for the `optionsbot init` CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from optionsbot.cli import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_init_creates_config_dir_and_writes_default_toml(
    runner: CliRunner, tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / "config"
    with patch(
        "optionsbot.cli._send_telegram_test", new=AsyncMock(return_value=None),
    ):
        result = runner.invoke(
            app,
            ["init", "--non-interactive", "--config-dir", str(cfg_dir)],
        )
    assert result.exit_code == 0, result.output
    cfg_path = cfg_dir / "config.toml"
    assert cfg_path.exists()
    body = cfg_path.read_text()
    assert "[ibkr]" in body
    assert "[telegram]" in body
    assert "[scan]" in body


def test_init_runs_alembic_migrations(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """init should create the SQLite DB at the configured path with the
    expected tables."""
    cfg_dir = tmp_path / "config"
    db_path = tmp_path / "data" / "optionsbot.db"
    monkeypatch.setenv("OPTIONSBOT_STORAGE__DB_PATH", str(db_path))

    with patch(
        "optionsbot.cli._send_telegram_test", new=AsyncMock(return_value=None),
    ):
        result = runner.invoke(
            app,
            ["init", "--non-interactive", "--config-dir", str(cfg_dir),
             "--skip-telegram"],
        )
    assert result.exit_code == 0, result.output
    assert db_path.exists()
    # Quick check: the watchlist table exists.
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='watchlist'"
    )
    assert cur.fetchone() is not None
    conn.close()


def test_init_skip_telegram_does_not_send_test(
    runner: CliRunner, tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / "config"
    send_mock = AsyncMock(return_value=12345)
    with patch("optionsbot.cli._send_telegram_test", new=send_mock):
        result = runner.invoke(
            app,
            ["init", "--non-interactive", "--config-dir", str(cfg_dir),
             "--skip-telegram"],
        )
    assert result.exit_code == 0, result.output
    send_mock.assert_not_called()


def test_init_idempotent_does_not_overwrite_existing_config(
    runner: CliRunner, tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    existing = cfg_dir / "config.toml"
    existing.write_text("# custom user config\n[ibkr]\nhost = \"my.server\"\n")
    with patch(
        "optionsbot.cli._send_telegram_test", new=AsyncMock(return_value=None),
    ):
        result = runner.invoke(
            app,
            ["init", "--non-interactive", "--config-dir", str(cfg_dir),
             "--skip-telegram"],
        )
    assert result.exit_code == 0, result.output
    # Custom marker preserved.
    assert "# custom user config" in existing.read_text()


def test_init_sends_telegram_test_when_creds_present(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_dir = tmp_path / "config"
    monkeypatch.setenv("OPTIONSBOT_TELEGRAM__BOT_TOKEN", "test-token")
    monkeypatch.setenv("OPTIONSBOT_TELEGRAM__CHAT_ID", "test-chat")
    send_mock = AsyncMock(return_value=99)
    with patch("optionsbot.cli._send_telegram_test", new=send_mock):
        result = runner.invoke(
            app,
            ["init", "--non-interactive", "--config-dir", str(cfg_dir)],
        )
    assert result.exit_code == 0, result.output
    send_mock.assert_awaited_once()


def test_init_prints_next_steps_summary(
    runner: CliRunner, tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / "config"
    with patch(
        "optionsbot.cli._send_telegram_test", new=AsyncMock(return_value=None),
    ):
        result = runner.invoke(
            app,
            ["init", "--non-interactive", "--config-dir", str(cfg_dir),
             "--skip-telegram"],
        )
    assert "next steps" in result.output.lower()
    assert "optionsbot status" in result.output
    assert "optionsbot-daemon" in result.output
