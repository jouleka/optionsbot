"""Shared fixtures for MCP server tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Engine

from optionsbot.config import Settings
from optionsbot.ibkr import IBKRClient
from optionsbot.mcp_server.context import ServerContext
from optionsbot.storage.db import create_engine_for_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _apply_migrations(db_path: Path) -> None:
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")


@pytest.fixture()
def mcp_engine(tmp_path: Path) -> Engine:
    """A SQLite engine pointed at a freshly-migrated DB in pytest's tmp_path."""
    db_path = tmp_path / "mcp.db"
    _apply_migrations(db_path)
    return create_engine_for_path(db_path)


@pytest.fixture()
def mcp_settings(tmp_path: Path) -> Settings:
    """Settings with the DB path pointed inside tmp_path so tests don't share state."""
    s = Settings()
    s.storage.db_path = tmp_path / "mcp.db"
    return s


@pytest.fixture()
def mock_ibkr_client() -> MagicMock:
    """An ``IBKRClient``-shaped mock. Async methods are AsyncMock; sync are MagicMock."""
    c = MagicMock(spec=IBKRClient)
    c.connect = AsyncMock()
    c.ensure_connected = AsyncMock()
    c.disconnect = AsyncMock()
    return c


@pytest.fixture()
def server_context(mcp_settings: Settings, mcp_engine: Engine) -> ServerContext:
    """A ServerContext with real engine + settings but no IBKRClient (lazy)."""
    return ServerContext(settings=mcp_settings, engine=mcp_engine)
