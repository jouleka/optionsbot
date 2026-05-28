"""Integration-test fixtures: real engine + IBKR-emulating mock."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine

from optionsbot.config import Settings
from optionsbot.storage.db import create_engine_for_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _apply_migrations(db_path: Path) -> None:
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")


@pytest.fixture()
def integration_engine(tmp_path: Path) -> Engine:
    db_path = tmp_path / "integration.db"
    _apply_migrations(db_path)
    return create_engine_for_path(db_path)


@pytest.fixture()
def integration_settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.storage.db_path = tmp_path / "integration.db"
    return s
