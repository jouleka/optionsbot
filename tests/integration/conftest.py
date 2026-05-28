"""Integration-test fixtures: real engine + IBKR-emulating mock.

Autouse fixtures isolate the test run from the user's home directory:
- HistoryClient's parquet cache is redirected to tmp_path so synthetic
  fixture bars don't poison the real ~/.cache/optionsbot/history/.
- next_earnings (yfinance lookup) is stubbed so the test has no network
  dependency and earnings logic is deterministic.
"""

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


@pytest.fixture(autouse=True)
def _isolate_history_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect HistoryClient's parquet cache to tmp_path.

    Without this, end-to-end scan_symbol tests write synthetic fixture
    bars into ~/.cache/optionsbot/history/<SYMBOL>-<DATE>.parquet --
    polluting the user's real cache directory and corrupting any
    subsequent real-daemon read for the same symbol and end_date.
    """
    cache_dir = tmp_path / "history_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "optionsbot.ibkr.history._DEFAULT_CACHE_DIR", cache_dir
    )


@pytest.fixture(autouse=True)
def _stub_next_earnings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace yfinance-backed next_earnings with a deterministic stub.

    Without this, every scan_symbol call in the integration tests fires
    a live yfinance HTTP request (caught by a blanket except but adds
    network dependency and surfaces 404 logs on every run).
    """
    from optionsbot.analysis.types import EarningsInfo

    def _no_earnings(symbol: str, manual_overrides=None):  # type: ignore[no-untyped-def]
        return EarningsInfo(next_date=None, source="manual")

    monkeypatch.setattr("optionsbot.analysis.view.next_earnings", _no_earnings)


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
