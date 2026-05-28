"""Tests for the `optionsbot status` CLI command."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import insert
from typer.testing import CliRunner

from optionsbot.cli import app
from optionsbot.storage.db import create_engine_for_path
from optionsbot.storage.schema import alerts, scan_runs

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _apply_migrations(db_path: Path) -> None:
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def configured_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up a tmp DB + env vars so optionsbot status finds a valid config."""
    db_path = tmp_path / "optionsbot.db"
    _apply_migrations(db_path)
    monkeypatch.setenv("OPTIONSBOT_STORAGE__DB_PATH", str(db_path))
    return db_path


def _fake_socket_ok(*args, **kwargs):
    """Mimic socket.create_connection's context-manager return."""
    sock = MagicMock()
    sock.__enter__ = MagicMock(return_value=sock)
    sock.__exit__ = MagicMock(return_value=False)
    return sock


def _fake_telegram_response(username: str = "test_bot") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"ok": True, "result": {"username": username}})
    return resp


def test_status_all_ok_exits_zero(
    runner: CliRunner,
    configured_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPTIONSBOT_TELEGRAM__BOT_TOKEN", "test-token")
    monkeypatch.setenv("OPTIONSBOT_TELEGRAM__CHAT_ID", "test-chat")
    with patch("socket.create_connection", side_effect=_fake_socket_ok):
        mock_resp = _fake_telegram_response("optionsbot_bot")
        async_client = MagicMock()
        async_client.__aenter__ = AsyncMock(return_value=async_client)
        async_client.__aexit__ = AsyncMock(return_value=False)
        async_client.get = AsyncMock(return_value=mock_resp)
        with patch("httpx.AsyncClient", return_value=async_client):
            result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert "✓ db" in result.output
    assert "✓ ibkr" in result.output
    assert "✓ telegram" in result.output
    assert "@optionsbot_bot" in result.output


def test_status_db_missing_exits_one(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Point storage at a non-existent DB.
    monkeypatch.setenv("OPTIONSBOT_STORAGE__DB_PATH", str(tmp_path / "missing.db"))
    monkeypatch.setenv("OPTIONSBOT_TELEGRAM__BOT_TOKEN", "")
    monkeypatch.setenv("OPTIONSBOT_TELEGRAM__CHAT_ID", "")
    with patch("socket.create_connection", side_effect=_fake_socket_ok):
        result = runner.invoke(app, ["status", "--no-telegram"])
    assert result.exit_code == 1
    assert "✗ db" in result.output


def test_status_ibkr_unreachable_exits_one(
    runner: CliRunner, configured_env: Path
) -> None:
    with patch("socket.create_connection", side_effect=ConnectionRefusedError("nope")):
        result = runner.invoke(app, ["status", "--no-telegram"])
    assert result.exit_code == 1
    assert "✗ ibkr" in result.output


def test_status_no_telegram_flag_skips_check(
    runner: CliRunner, configured_env: Path
) -> None:
    with patch("socket.create_connection", side_effect=_fake_socket_ok):
        result = runner.invoke(app, ["status", "--no-telegram"])
    assert result.exit_code == 0
    assert "skipped" in result.output


def test_status_last_scan_warn_when_old(
    runner: CliRunner,
    configured_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Insert a scan_runs row from 2 hours ago; with default interval_minutes=15
    and threshold 2x = 30m, it should be warn."""
    engine = create_engine_for_path(configured_env)
    old = datetime.now(UTC) - timedelta(hours=2)
    with engine.begin() as conn:
        conn.execute(
            insert(scan_runs).values(
                started=old,
                finished=old,
                tickers_scanned=1,
                alerts_fired=0,
            )
        )
    with patch("socket.create_connection", side_effect=_fake_socket_ok):
        result = runner.invoke(app, ["status", "--no-telegram"])
    assert result.exit_code == 0  # not a critical failure
    assert "⚠ last scan" in result.output


def test_status_last_alert_shows_recent_sent(
    runner: CliRunner,
    configured_env: Path,
) -> None:
    engine = create_engine_for_path(configured_env)
    recent = datetime.now(UTC) - timedelta(minutes=5)
    with engine.begin() as conn:
        conn.execute(
            insert(alerts).values(
                ts=recent,
                symbol="AAPL",
                strategy="iron_condor",
                score=85.0,
                status="sent",
            )
        )
    with patch("socket.create_connection", side_effect=_fake_socket_ok):
        result = runner.invoke(app, ["status", "--no-telegram"])
    assert "✓ last alert" in result.output


def test_status_json_output(
    runner: CliRunner,
    configured_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPTIONSBOT_TELEGRAM__BOT_TOKEN", "")
    monkeypatch.setenv("OPTIONSBOT_TELEGRAM__CHAT_ID", "")
    with patch("socket.create_connection", side_effect=_fake_socket_ok):
        result = runner.invoke(app, ["status", "--no-telegram", "--json"])
    # Output should be parseable JSON list of dicts.
    parsed = json.loads(result.output)
    assert isinstance(parsed, list)
    names = {entry["name"] for entry in parsed}
    assert "db" in names
    assert "ibkr" in names
    assert "telegram" in names
