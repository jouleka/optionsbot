"""Shared fixtures for daemon tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Engine

from optionsbot.config import Settings
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.telegram_client import TelegramClient
from optionsbot.ibkr import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.storage.db import create_engine_for_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _apply_migrations(db_path: Path) -> None:
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")


@pytest.fixture()
def daemon_engine(tmp_path: Path) -> Engine:
    db_path = tmp_path / "daemon.db"
    _apply_migrations(db_path)
    return create_engine_for_path(db_path)


@pytest.fixture()
def daemon_settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.storage.db_path = tmp_path / "daemon.db"
    s.telegram.bot_token = "test-token"
    s.telegram.chat_id = "test-chat"
    return s


@pytest.fixture()
def mock_ibkr_client() -> MagicMock:
    c = MagicMock(spec=IBKRClient)
    c.connect = AsyncMock()
    c.ensure_connected = AsyncMock()
    c.disconnect = AsyncMock()
    return c


@pytest.fixture()
def mock_telegram() -> MagicMock:
    tg = MagicMock(spec=TelegramClient)
    tg.send_message = AsyncMock(return_value=12345)
    tg.aclose = AsyncMock()
    return tg


@pytest.fixture()
def daemon_context(
    daemon_settings: Settings,
    daemon_engine: Engine,
    mock_ibkr_client: MagicMock,
    mock_telegram: MagicMock,
) -> DaemonContext:
    resolver = ContractResolver(mock_ibkr_client)
    return DaemonContext(
        settings=daemon_settings,
        engine=daemon_engine,
        ibkr=mock_ibkr_client,
        resolver=resolver,
        telegram=mock_telegram,
    )
