"""Fixtures for scan-helper tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from sqlalchemy import Engine

from optionsbot.config import Settings
from optionsbot.ibkr import IBKRClient
from optionsbot.ibkr.types import AccountSummary, OptionChainLeg, StockQuote
from optionsbot.storage.db import create_engine_for_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _apply_migrations(db_path: Path) -> None:
    from alembic.config import Config

    from alembic import command

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")


@pytest.fixture()
def scan_engine(tmp_path: Path) -> Engine:
    db_path = tmp_path / "scan.db"
    _apply_migrations(db_path)
    return create_engine_for_path(db_path)


@pytest.fixture()
def scan_settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.storage.db_path = tmp_path / "scan.db"
    return s


@pytest.fixture()
def fake_bars() -> pd.DataFrame:
    """120 trading days of synthetic OHLCV bars for SPY @ ~$400."""
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(120)]
    closes = [400.0 + i * 0.1 for i in range(120)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1_000_000] * 120,
        },
        index=pd.Index(dates, name="date"),
    )


def _chain_leg(
    expiry: str,
    strike: float,
    right: str,
    *,
    iv: float = 0.20,
    delta: float | None = None,
) -> OptionChainLeg:
    if delta is None:
        delta = 0.16 if right == "C" else -0.16
    return OptionChainLeg(
        symbol="SPY",
        expiry=expiry,
        strike=strike,
        right=right,  # type: ignore[arg-type]
        bid=1.0,
        ask=1.1,
        iv=iv,
        delta=delta,
        gamma=0.01,
        theta=-0.02,
        vega=0.1,
        open_interest=1000,
        volume=50,
    )


@pytest.fixture()
def fake_chain() -> list[OptionChainLeg]:
    expiry = (date.today() + timedelta(days=45)).strftime("%Y%m%d")
    return [
        _chain_leg(expiry, strike, right)
        for strike in (385, 390, 395, 400, 405, 410, 415)
        for right in ("C", "P")
    ]


@pytest.fixture()
def fake_stock_quote() -> StockQuote:
    return StockQuote(
        symbol="SPY",
        bid=399.9,
        ask=400.1,
        last=400.0,
        mid=400.0,
        ts=datetime.now(UTC),
        delayed=True,
    )


@pytest.fixture()
def mock_ibkr_for_scan(
    monkeypatch: pytest.MonkeyPatch,
    fake_bars: pd.DataFrame,
    fake_chain: list[OptionChainLeg],
    fake_stock_quote: StockQuote,
) -> MagicMock:
    """Mock IBKRClient + the four sub-clients scan_symbol constructs internally."""
    ibkr = MagicMock(spec=IBKRClient)
    ibkr.ensure_connected = AsyncMock()
    ibkr.settings = Settings()

    import optionsbot.scan.symbol as symbol_mod

    history_mock = MagicMock()
    history_mock.get_history = AsyncMock(return_value=fake_bars)
    monkeypatch.setattr(symbol_mod, "HistoryClient", MagicMock(return_value=history_mock))

    chain_mock = MagicMock()
    chain_mock.get_chain = AsyncMock(return_value=fake_chain)
    monkeypatch.setattr(symbol_mod, "ChainClient", MagicMock(return_value=chain_mock))

    market_mock = MagicMock()
    market_mock.get_stock_snapshot = AsyncMock(return_value=fake_stock_quote)
    monkeypatch.setattr(symbol_mod, "MarketDataClient", MagicMock(return_value=market_mock))

    positions_mock = MagicMock()
    positions_mock.get_positions = AsyncMock(return_value=[])
    positions_mock.get_account_summary = AsyncMock(
        return_value=AccountSummary(
            net_liquidation=Decimal("100000"),
            buying_power=Decimal("100000"),
            available_funds=Decimal("100000"),
            currency="USD",
        )
    )
    monkeypatch.setattr(symbol_mod, "PositionsClient", MagicMock(return_value=positions_mock))

    return ibkr


@pytest.fixture(autouse=True)
def _stub_news_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """scan_symbol calls refresh_news_if_stale (yfinance) -- stub it offline."""
    import optionsbot.scan.symbol as symbol_mod

    monkeypatch.setattr(
        symbol_mod, "refresh_news_if_stale", lambda *a, **k: None, raising=False
    )
