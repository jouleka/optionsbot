"""Tests for the four watchlist MCP tools (IBK-51..53, IBK-55)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import insert, select

from optionsbot.mcp_server.context import ServerContext
from optionsbot.mcp_server.tools.watchlist import register
from optionsbot.storage.schema import snapshots, watchlist


class _FakeCtx:
    """Minimal stand-in for FastMCP's Context. Tools only touch
    ``ctx.request_context.lifespan_context``."""

    def __init__(self, lifespan_context: ServerContext) -> None:
        rc = MagicMock()
        rc.lifespan_context = lifespan_context
        self.request_context = rc


def _get_tools(server_context: ServerContext) -> dict:
    """Build a throwaway FastMCP, register watchlist tools, return them by name."""
    from mcp.server.fastmcp import FastMCP

    captured: dict[str, object] = {}

    class _Capture(FastMCP):
        def tool(self, *a, **kw):  # type: ignore[override]
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    server = _Capture("test")
    register(server)
    return captured


# ---- add_to_watchlist (IBK-51) --------------------------------------------

async def test_add_to_watchlist_persists_and_validates(
    server_context: ServerContext, mock_ibkr_client: MagicMock
) -> None:
    server_context._ibkr = mock_ibkr_client
    tools = _get_tools(server_context)
    add = tools["add_to_watchlist"]

    with patch(
        "optionsbot.mcp_server.tools.watchlist.ContractResolver"
    ) as ResolverCls:
        resolver = MagicMock()
        resolver.stock = AsyncMock()
        ResolverCls.return_value = resolver
        result = await add(symbol="aapl", notes=None, ctx=_FakeCtx(server_context))

    assert result["ok"] is True
    assert result["symbol"] == "AAPL"
    assert "added_at" in result
    with server_context.engine.connect() as conn:
        rows = conn.execute(select(watchlist)).fetchall()
    assert len(rows) == 1
    assert rows[0].symbol == "AAPL"


async def test_add_to_watchlist_idempotent(
    server_context: ServerContext, mock_ibkr_client: MagicMock
) -> None:
    server_context._ibkr = mock_ibkr_client
    with server_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="MSFT", added_at=datetime.now(UTC)))
    tools = _get_tools(server_context)
    add = tools["add_to_watchlist"]

    with patch(
        "optionsbot.mcp_server.tools.watchlist.ContractResolver"
    ) as ResolverCls:
        resolver = MagicMock()
        resolver.stock = AsyncMock()
        ResolverCls.return_value = resolver
        result = await add(symbol="MSFT", notes=None, ctx=_FakeCtx(server_context))

    assert result == {"ok": True, "symbol": "MSFT", "already_present": True}


async def test_add_to_watchlist_rejects_unknown_symbol(
    server_context: ServerContext, mock_ibkr_client: MagicMock
) -> None:
    server_context._ibkr = mock_ibkr_client
    tools = _get_tools(server_context)
    add = tools["add_to_watchlist"]

    with patch(
        "optionsbot.mcp_server.tools.watchlist.ContractResolver"
    ) as ResolverCls:
        resolver = MagicMock()
        resolver.stock = AsyncMock(side_effect=ValueError("Could not qualify"))
        ResolverCls.return_value = resolver
        result = await add(symbol="ZZZZ", notes=None, ctx=_FakeCtx(server_context))

    assert result["ok"] is False
    assert result["error"] == "unknown_symbol"


async def test_add_to_watchlist_handles_ibkr_unavailable(
    server_context: ServerContext, mock_ibkr_client: MagicMock
) -> None:
    server_context._ibkr = mock_ibkr_client
    tools = _get_tools(server_context)
    add = tools["add_to_watchlist"]

    with patch(
        "optionsbot.mcp_server.tools.watchlist.ContractResolver"
    ) as ResolverCls:
        resolver = MagicMock()
        resolver.stock = AsyncMock(side_effect=ConnectionError("gateway down"))
        ResolverCls.return_value = resolver
        result = await add(symbol="AAPL", notes=None, ctx=_FakeCtx(server_context))

    assert result["ok"] is False
    assert result["error"] == "ibkr_unavailable"


async def test_add_to_watchlist_rejects_empty_symbol(
    server_context: ServerContext,
) -> None:
    tools = _get_tools(server_context)
    add = tools["add_to_watchlist"]
    result = await add(symbol="   ", notes=None, ctx=_FakeCtx(server_context))
    assert result == {"ok": False, "error": "invalid_symbol", "message": "symbol is empty"}


# ---- remove_from_watchlist (IBK-52) ---------------------------------------

async def test_remove_from_watchlist_deletes_row(
    server_context: ServerContext,
) -> None:
    with server_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=datetime.now(UTC)))
    tools = _get_tools(server_context)
    remove = tools["remove_from_watchlist"]

    result = await remove(symbol="AAPL", ctx=_FakeCtx(server_context))
    assert result == {"ok": True, "symbol": "AAPL", "removed": True}
    with server_context.engine.connect() as conn:
        assert conn.execute(select(watchlist)).fetchall() == []


async def test_remove_from_watchlist_unknown_symbol_returns_removed_false(
    server_context: ServerContext,
) -> None:
    tools = _get_tools(server_context)
    remove = tools["remove_from_watchlist"]
    result = await remove(symbol="NOPE", ctx=_FakeCtx(server_context))
    assert result == {"ok": True, "symbol": "NOPE", "removed": False}


async def test_remove_from_watchlist_preserves_snapshots(
    server_context: ServerContext,
) -> None:
    """Per IBK-52: removing a ticker keeps its snapshot history."""
    now = datetime.now(UTC)
    with server_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=now))
        conn.execute(insert(snapshots).values(symbol="AAPL", ts=now))
    tools = _get_tools(server_context)
    remove = tools["remove_from_watchlist"]
    await remove(symbol="AAPL", ctx=_FakeCtx(server_context))
    with server_context.engine.connect() as conn:
        snaps = conn.execute(select(snapshots)).fetchall()
    assert len(snaps) == 1, "snapshot history must survive watchlist removal"


# ---- list_watchlist (IBK-53) ----------------------------------------------

async def test_list_watchlist_empty(server_context: ServerContext) -> None:
    tools = _get_tools(server_context)
    listfn = tools["list_watchlist"]
    result = await listfn(ctx=_FakeCtx(server_context))
    assert result == {"ok": True, "entries": []}


async def test_list_watchlist_returns_all_with_overrides_and_last_scanned(
    server_context: ServerContext,
) -> None:
    older = datetime(2026, 5, 26, tzinfo=UTC)
    newer = datetime(2026, 5, 27, 10, 0, tzinfo=UTC)
    with server_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(
            symbol="AAPL", added_at=older, view_override_dir="bull"
        ))
        conn.execute(insert(watchlist).values(
            symbol="MSFT", added_at=older, notes="cloud bet"
        ))
        conn.execute(insert(snapshots).values(symbol="AAPL", ts=older))
        conn.execute(insert(snapshots).values(symbol="AAPL", ts=newer))
    tools = _get_tools(server_context)
    listfn = tools["list_watchlist"]
    result = await listfn(ctx=_FakeCtx(server_context))

    assert result["ok"] is True
    by_sym = {e["symbol"]: e for e in result["entries"]}
    assert by_sym["AAPL"]["view_override"] == {"direction": "bull", "iv_regime": None}
    assert by_sym["AAPL"]["last_scanned"] == newer.isoformat()
    assert by_sym["MSFT"]["last_scanned"] is None
    assert by_sym["MSFT"]["notes"] == "cloud bet"


# ---- set_view_override (IBK-55) -------------------------------------------

async def test_set_view_override_sets_direction(
    server_context: ServerContext,
) -> None:
    with server_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=datetime.now(UTC)))
    tools = _get_tools(server_context)
    setfn = tools["set_view_override"]
    result = await setfn(
        symbol="AAPL", direction="bull", iv_regime=None, ctx=_FakeCtx(server_context)
    )
    assert result["ok"] is True
    assert result["override"] == {"direction": "bull", "iv_regime": None}


async def test_set_view_override_clears_with_none(
    server_context: ServerContext,
) -> None:
    with server_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(
            symbol="AAPL", added_at=datetime.now(UTC),
            view_override_dir="bull", view_override_iv="high",
        ))
    tools = _get_tools(server_context)
    setfn = tools["set_view_override"]
    result = await setfn(
        symbol="AAPL", direction=None, iv_regime=None, ctx=_FakeCtx(server_context)
    )
    assert result["override"] == {"direction": None, "iv_regime": None}


async def test_set_view_override_rejects_invalid_direction(
    server_context: ServerContext,
) -> None:
    with server_context.engine.begin() as conn:
        conn.execute(insert(watchlist).values(symbol="AAPL", added_at=datetime.now(UTC)))
    tools = _get_tools(server_context)
    setfn = tools["set_view_override"]
    result = await setfn(
        symbol="AAPL", direction="moonshot", iv_regime=None, ctx=_FakeCtx(server_context)
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_direction"


async def test_set_view_override_unknown_symbol(
    server_context: ServerContext,
) -> None:
    tools = _get_tools(server_context)
    setfn = tools["set_view_override"]
    result = await setfn(
        symbol="NOPE", direction="bull", iv_regime=None, ctx=_FakeCtx(server_context)
    )
    assert result["ok"] is False
    assert result["error"] == "unknown_symbol"
