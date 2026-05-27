"""Tests for the lifespan-scoped ServerContext."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from optionsbot.ibkr import IBKRClient
from optionsbot.mcp_server.context import ServerContext, app_lifespan


async def test_ibkr_is_lazy(server_context: ServerContext) -> None:
    """Constructing a ServerContext does NOT create an IBKRClient."""
    assert server_context._ibkr is None


async def test_ibkr_constructs_on_demand(server_context: ServerContext) -> None:
    """First call to ibkr() instantiates and returns an IBKRClient."""
    client = await server_context.ibkr()
    assert isinstance(client, IBKRClient)
    assert client.role == "mcp"
    # Second call returns the same instance.
    again = await server_context.ibkr()
    assert again is client


async def test_shutdown_disconnects_ibkr_when_present(
    server_context: ServerContext,
) -> None:
    """shutdown() disconnects the IBKR client if one was constructed."""
    fake = MagicMock(spec=IBKRClient)
    fake.disconnect = AsyncMock()
    server_context._ibkr = fake
    await server_context.shutdown()
    fake.disconnect.assert_awaited_once()


async def test_shutdown_skips_ibkr_when_never_used(
    server_context: ServerContext,
) -> None:
    """shutdown() is a no-op for IBKR when ibkr() was never called."""
    # Should not raise. engine.dispose() is idempotent on a real engine.
    await server_context.shutdown()


async def test_app_lifespan_yields_context_with_engine_and_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """app_lifespan yields a ServerContext with engine + settings, tears down cleanly."""
    from optionsbot.config import get_settings

    # Point storage at tmp_path before lifespan starts.
    get_settings.cache_clear()
    monkeypatch.setenv("OPTIONSBOT_STORAGE__DB_PATH", str(tmp_path / "lifespan.db"))
    # Pre-migrate the DB so create_engine_for_path doesn't try to migrate it.
    from tests.unit.test_mcp.conftest import _apply_migrations  # type: ignore[import-not-found]
    _apply_migrations(tmp_path / "lifespan.db")

    async with app_lifespan(MagicMock()) as ctx:
        assert isinstance(ctx, ServerContext)
        assert ctx.settings.storage.db_path == tmp_path / "lifespan.db"
        assert ctx.engine is not None
    # After exit, engine is disposed -- a connection attempt should still work
    # (dispose drops the pool but lets new connections open) so we just assert
    # no exception was raised.

    get_settings.cache_clear()
