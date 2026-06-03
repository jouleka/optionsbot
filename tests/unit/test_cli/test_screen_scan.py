"""Tests for `optionsbot screen --scan` (IBK-95 Phase B)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from optionsbot.cli import app
from optionsbot.storage.db import create_engine_for_path
from optionsbot.storage.schema import scan_runs

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


def test_screen_scan_runs_records_and_passes_scan_top(runner: CliRunner, db: Path) -> None:
    cand = MagicMock(symbol="SPY", hv_rank=0.81, dollar_volume=1.2e9)
    result_obj = MagicMock(scored=())  # no strategies above threshold
    fake_client = MagicMock()
    fake_client.connect = AsyncMock()
    fake_client.disconnect = AsyncMock()
    fake_sns = AsyncMock(return_value=[(cand, result_obj)])

    with (
        patch("optionsbot.ibkr.IBKRClient", return_value=fake_client),
        patch("optionsbot.screener.screen.screen_and_scan", fake_sns),
    ):
        result = runner.invoke(app, ["screen", "--scan", "--scan-top", "2"])

    assert result.exit_code == 0, result.output
    fake_sns.assert_awaited_once()
    assert fake_sns.await_args.kwargs["scan_top_n"] == 2  # --scan-top override flows through
    assert "SPY" in result.output

    engine = create_engine_for_path(db)
    with engine.connect() as conn:
        rows = conn.execute(select(scan_runs.c.tickers_scanned)).fetchall()
    assert len(rows) == 1
    assert rows[0].tickers_scanned == 1
