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


def test_screen_scan_renders_picks_and_records_errors(runner: CliRunner, db: Path) -> None:
    """Real screen_and_scan + top_k: a >=70 pick renders; a failed Stage-2 scan
    is skipped but recorded in scan_runs.errors_json."""
    from optionsbot.screener.screen import ScreenCandidate

    cand_spy = ScreenCandidate(symbol="SPY", hv_rank=0.82, dollar_volume=1e9)
    cand_aapl = ScreenCandidate(symbol="AAPL", hv_rank=0.71, dollar_volume=5e8)

    async def fake_screen_universe(hc, uni, mdv):
        return (cand_spy, cand_aapl)

    pick = MagicMock(strategy_name="bull_put_spread", score=85.0)
    pick.suggestion.expected_value = 50.0   # positive edge -> no banner
    pick.suggestion.risk_normalized_expectancy = 0.05
    spy_result = MagicMock(scored=(pick,))

    async def fake_scan_symbol(symbol, *args, **kwargs):
        if symbol == "AAPL":
            raise RuntimeError("boom")
        return spy_result

    fake_client = MagicMock()
    fake_client.connect = AsyncMock()
    fake_client.disconnect = AsyncMock()

    with (
        patch("optionsbot.ibkr.IBKRClient", return_value=fake_client),
        patch("optionsbot.screener.screen.screen_universe", fake_screen_universe),
        patch("optionsbot.scan.scan_symbol", fake_scan_symbol),
    ):
        result = runner.invoke(app, ["screen", "--scan", "--scan-top", "2"])

    assert result.exit_code == 0, result.output
    assert "bull_put_spread: 85" in result.output  # the >=70 pick renders
    assert "SPY" in result.output

    engine = create_engine_for_path(db)
    with engine.connect() as conn:
        rows = conn.execute(
            select(scan_runs.c.tickers_scanned, scan_runs.c.errors_json)
        ).fetchall()
    assert len(rows) == 1
    assert rows[0].tickers_scanned == 1  # only SPY succeeded
    assert rows[0].errors_json is not None
    assert any("AAPL" in e for e in rows[0].errors_json)


def test_plain_screen_writes_no_scan_runs(runner: CliRunner, db: Path) -> None:
    """The plain `screen` (no --scan) path is unchanged: prints the ranking,
    writes no scan_runs row."""
    from optionsbot.screener.screen import ScreenCandidate

    async def fake_screen_universe(hc, uni, mdv):
        return (ScreenCandidate(symbol="SPY", hv_rank=0.8, dollar_volume=1e9),)

    fake_client = MagicMock()
    fake_client.connect = AsyncMock()
    fake_client.disconnect = AsyncMock()

    with (
        patch("optionsbot.ibkr.IBKRClient", return_value=fake_client),
        patch("optionsbot.screener.screen.screen_universe", fake_screen_universe),
    ):
        result = runner.invoke(app, ["screen"])

    assert result.exit_code == 0, result.output
    assert "SPY" in result.output
    engine = create_engine_for_path(db)
    with engine.connect() as conn:
        rows = conn.execute(select(scan_runs.c.id)).fetchall()
    assert rows == []  # plain screen writes no scan_runs heartbeat


def test_screen_scan_warns_when_no_positive_edge(runner: CliRunner, db: Path) -> None:
    from optionsbot.screener.screen import ScreenCandidate

    cand_spy = ScreenCandidate(symbol="SPY", hv_rank=0.82, dollar_volume=1e9)

    async def fake_screen_universe(hc, uni, mdv):
        return (cand_spy,)

    pick = MagicMock(strategy_name="bull_put_spread", score=85.0)
    pick.suggestion.expected_value = -50.0
    pick.suggestion.max_loss = 1000.0
    pick.suggestion.risk_normalized_expectancy = -0.05
    spy_result = MagicMock(scored=(pick,))

    async def fake_scan_symbol(symbol, *args, **kwargs):
        return spy_result

    fake_client = MagicMock()
    fake_client.connect = AsyncMock()
    fake_client.disconnect = AsyncMock()

    with (
        patch("optionsbot.ibkr.IBKRClient", return_value=fake_client),
        patch("optionsbot.screener.screen.screen_universe", fake_screen_universe),
        patch("optionsbot.scan.scan_symbol", fake_scan_symbol),
    ):
        result = runner.invoke(app, ["screen", "--scan", "--scan-top", "1"])

    assert result.exit_code == 0, result.output
    assert "No positive-edge" in result.output
    assert "bull_put_spread: 85" in result.output
