"""Live paper IB Gateway smoke test (IBK-71).

@pytest.mark.live -- skipped by default. To run:

    uv run pytest tests/integration/test_live_ibkr.py -m live -v

Requires:
- IB Gateway running on the configured port (default 4002 / paper).
- Network reachable from WSL to the gateway.

This test validates the end-to-end production path against the real
ib_async library + a real (paper) IB Gateway -- catches integration
bugs that the mock-based suite necessarily misses. It is NOT run in
regular CI; manual / nightly only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from optionsbot.config import Settings
from optionsbot.ibkr import IBKRClient
from optionsbot.ibkr.chains import ChainClient
from optionsbot.scan import scan_symbol
from optionsbot.storage.db import create_engine_for_path


@pytest.mark.live
async def test_live_ibkr_scan_spy(tmp_path: Path) -> None:
    """Connect to paper IB Gateway, fetch SPY chain, run scan_symbol,
    assert at least one scored strategy + clean disconnect."""
    # Use default settings (host/port/client_id_mcp from config or env).
    settings = Settings()
    db_path = tmp_path / "live.db"

    # Apply migrations.
    from alembic.config import Config

    from alembic import command

    project_root = Path(__file__).resolve().parents[2]
    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")
    settings.storage.db_path = db_path

    engine = create_engine_for_path(db_path)
    client = IBKRClient(role="cli", settings=settings)

    try:
        # 1. Connect.
        await client.connect()
        assert client.ib.isConnected(), "expected IB Gateway connection"

        # 2. Fetch chain directly via ChainClient -- standalone check before
        # scan_symbol uses it internally.
        chain_client = ChainClient(client)
        chain = await chain_client.get_chain("SPY")
        assert len(chain) >= 1, "expected at least one leg in the SPY chain"

        # 3. Full end-to-end scan.
        result = await scan_symbol(
            "SPY",
            client,
            engine,
            settings,
        )
        assert result.symbol == "SPY"
        assert result.snapshot_id > 0
        assert result.snapshot_ts is not None
        assert result.view is not None
        # At least one strategy should be applicable to the real-world snapshot
        # (across 16 strategies the live view should match at least one set
        # of applicable_views).
        assert len(result.scored) >= 1, (
            "expected at least one scored strategy for SPY; got 0 -- check "
            "that the chain returned IV data and that the inferred view "
            "matches at least one strategy's applicable_views"
        )
    finally:
        # Always disconnect even on assertion failure.
        await client.disconnect()
