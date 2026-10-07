"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine

from optionsbot.storage.db import create_engine_for_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def apply_migrations(db_path: Path) -> None:
    """Run alembic upgrade head against db_path using the Alembic Python API.

    The alembic env.py honors a sqlalchemy.url override set on the Config
    object, so we can point at any path without touching environment variables.
    """
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Engine:
    """A SQLite engine pointed at a freshly-migrated DB in pytest's tmp_path."""
    db_path = tmp_path / "test.db"
    apply_migrations(db_path)
    return create_engine_for_path(db_path)


@pytest.fixture(autouse=True)
def isolated_default_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Never load a developer's broker credentials or trading settings during tests."""
    from optionsbot import config

    monkeypatch.setattr(config, "DEFAULT_CONFIG_FILE", tmp_path / "isolated-config.toml")
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()
