"""Tests for the `optionsbot scan-once` CLI command (IBK-7)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import insert, select
from typer.testing import CliRunner

from optionsbot.cli import app
from optionsbot.storage.db import create_engine_for_path
from optionsbot.storage.schema import scan_runs, watchlist

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
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "optionsbot.db"
    _apply_migrations(db_path)
    monkeypatch.setenv("OPTIONSBOT_STORAGE__DB_PATH", str(db_path))
    return db_path


def test_scan_once_empty_watchlist_is_noop(runner: CliRunner, db: Path) -> None:
    result = runner.invoke(app, ["scan-once"])
    assert result.exit_code == 0, result.output
    assert "empty" in result.output.lower()


def test_scan_once_scans_watchlist_and_records_run(
    runner: CliRunner, db: Path
) -> None:
    engine = create_engine_for_path(db)
    with engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="SPY", added_at=datetime.now(UTC)))

    fake_client = MagicMock()
    fake_client.connect = AsyncMock()
    fake_client.disconnect = AsyncMock()
    fake_result = MagicMock()
    fake_result.scored = ()  # no strategies above threshold -> simplest path
    fake_scan = AsyncMock(return_value=fake_result)

    with (
        patch("optionsbot.ibkr.IBKRClient", return_value=fake_client),
        patch("optionsbot.scan.scan_symbol", fake_scan),
    ):
        result = runner.invoke(app, ["scan-once"])

    assert result.exit_code == 0, result.output
    fake_scan.assert_awaited_once()
    # A scan_runs heartbeat row should have been recorded.
    with engine.connect() as conn:
        rows = conn.execute(
            select(scan_runs.c.tickers_scanned, scan_runs.c.alerts_fired)
        ).fetchall()
    assert len(rows) == 1
    assert rows[0].tickers_scanned == 1
