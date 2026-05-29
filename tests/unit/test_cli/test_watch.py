"""Tests for the `optionsbot watch` CLI commands (IBK-51..53)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import insert, select
from typer.testing import CliRunner

from optionsbot.cli import app
from optionsbot.storage.db import create_engine_for_path
from optionsbot.storage.schema import watchlist

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


def _seed(db_path: Path, symbol: str) -> None:
    engine = create_engine_for_path(db_path)
    with engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol=symbol, added_at=datetime.now(UTC)))


def _symbols(db_path: Path) -> list[str]:
    engine = create_engine_for_path(db_path)
    with engine.connect() as conn:
        return [r.symbol for r in conn.execute(select(watchlist.c.symbol)).fetchall()]


def _mock_ibkr(stock_side_effect: object = None):
    """Return (patches) for IBKRClient + ContractResolver so no gateway is needed."""
    fake_client = MagicMock()
    fake_client.connect = AsyncMock()
    fake_client.disconnect = AsyncMock()
    fake_resolver = MagicMock()
    if stock_side_effect is not None:
        fake_resolver.stock = AsyncMock(side_effect=stock_side_effect)
    else:
        fake_resolver.stock = AsyncMock(return_value=MagicMock())
    return (
        patch("optionsbot.ibkr.IBKRClient", return_value=fake_client),
        patch("optionsbot.ibkr.contracts.ContractResolver", return_value=fake_resolver),
    )


def test_watch_list_shows_symbols(runner: CliRunner, db: Path) -> None:
    _seed(db, "SPY")
    _seed(db, "AAPL")
    result = runner.invoke(app, ["watch", "list"])
    assert result.exit_code == 0, result.output
    assert "SPY" in result.output
    assert "AAPL" in result.output


def test_watch_list_empty(runner: CliRunner, db: Path) -> None:
    result = runner.invoke(app, ["watch", "list"])
    assert result.exit_code == 0, result.output
    assert "empty" in result.output.lower()


def test_watch_remove_deletes_symbol(runner: CliRunner, db: Path) -> None:
    _seed(db, "SPY")
    result = runner.invoke(app, ["watch", "remove", "SPY"])
    assert result.exit_code == 0, result.output
    assert _symbols(db) == []


def test_watch_remove_missing_symbol_is_graceful(runner: CliRunner, db: Path) -> None:
    result = runner.invoke(app, ["watch", "remove", "NOPE"])
    assert result.exit_code == 0, result.output
    assert "not in" in result.output.lower()


def test_watch_add_inserts_after_ibkr_validation(runner: CliRunner, db: Path) -> None:
    p_client, p_resolver = _mock_ibkr()
    with p_client, p_resolver:
        result = runner.invoke(app, ["watch", "add", "spy"])  # lowercase -> stored upper
    assert result.exit_code == 0, result.output
    assert "SPY" in result.output
    assert _symbols(db) == ["SPY"]


def test_watch_add_rejects_unknown_symbol(runner: CliRunner, db: Path) -> None:
    p_client, p_resolver = _mock_ibkr(stock_side_effect=ValueError("Could not qualify"))
    with p_client, p_resolver:
        result = runner.invoke(app, ["watch", "add", "NOTREAL"])
    assert result.exit_code != 0
    assert _symbols(db) == []  # not inserted


def test_watch_add_idempotent(runner: CliRunner, db: Path) -> None:
    _seed(db, "SPY")
    p_client, p_resolver = _mock_ibkr()
    with p_client, p_resolver:
        result = runner.invoke(app, ["watch", "add", "SPY"])
    assert result.exit_code == 0, result.output
    assert _symbols(db) == ["SPY"]  # still single row
    assert "already" in result.output.lower()
