"""Tests for IBKRClient connect/ensure_connected/disconnect lifecycle."""

from __future__ import annotations

import pytest

from optionsbot.config import Settings
from optionsbot.ibkr.client import IBKRClient


def _settings_with_paper(paper: bool, port: int = 4002) -> Settings:
    s = Settings()
    s.ibkr.paper = paper
    s.ibkr.port = port
    return s


async def test_connect_uses_configured_port_and_client_id(mock_ib) -> None:
    s = _settings_with_paper(paper=True, port=4002)
    s.ibkr.client_id_daemon = 7
    client = IBKRClient(role="daemon", settings=s, ib=mock_ib, backoff_seconds=())
    mock_ib.isConnected.return_value = False
    await client.connect()
    mock_ib.connectAsync.assert_awaited_once()
    kwargs = mock_ib.connectAsync.await_args.kwargs
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 4002
    assert kwargs["clientId"] == 7


async def test_connect_sets_delayed_data_in_paper_mode(mock_ib) -> None:
    client = IBKRClient(
        role="cli",
        settings=_settings_with_paper(paper=True),
        ib=mock_ib,
        backoff_seconds=(),
    )
    mock_ib.isConnected.return_value = False
    await client.connect()
    mock_ib.reqMarketDataType.assert_called_once_with(3)


async def test_connect_does_not_force_delayed_in_live_mode(mock_ib) -> None:
    client = IBKRClient(
        role="cli",
        settings=_settings_with_paper(paper=False),
        ib=mock_ib,
        backoff_seconds=(),
    )
    mock_ib.isConnected.return_value = False
    await client.connect()
    mock_ib.reqMarketDataType.assert_not_called()


async def test_connect_is_noop_when_already_connected(mock_ib) -> None:
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    mock_ib.isConnected.return_value = True
    await client.connect()
    mock_ib.connectAsync.assert_not_awaited()


async def test_ensure_connected_calls_connect_when_disconnected(mock_ib) -> None:
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    mock_ib.isConnected.return_value = False
    await client.ensure_connected()
    mock_ib.connectAsync.assert_awaited_once()


async def test_ensure_connected_is_noop_when_connected(mock_ib) -> None:
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    mock_ib.isConnected.return_value = True
    await client.ensure_connected()
    mock_ib.connectAsync.assert_not_awaited()


async def test_disconnect_calls_underlying_disconnect(mock_ib) -> None:
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    mock_ib.isConnected.return_value = True
    await client.disconnect()
    mock_ib.disconnect.assert_called_once()


async def test_reconnect_with_backoff_eventually_succeeds(mock_ib) -> None:
    # First two connectAsync attempts fail, third succeeds.
    mock_ib.isConnected.return_value = False
    mock_ib.connectAsync.side_effect = [
        ConnectionError("attempt 1"),
        ConnectionError("attempt 2"),
        None,
    ]
    client = IBKRClient(
        role="cli",
        settings=Settings(),
        ib=mock_ib,
        # zero-second backoffs so the test is fast
        backoff_seconds=(0, 0),
    )
    await client.connect()
    assert mock_ib.connectAsync.await_count == 3


async def test_reconnect_eventually_gives_up(mock_ib) -> None:
    mock_ib.isConnected.return_value = False
    mock_ib.connectAsync.side_effect = ConnectionError("always fails")
    client = IBKRClient(
        role="cli",
        settings=Settings(),
        ib=mock_ib,
        backoff_seconds=(0, 0),  # 1 initial attempt + 2 retries
    )
    with pytest.raises(ConnectionError, match="Could not connect"):
        await client.connect()
    assert mock_ib.connectAsync.await_count == 3


async def test_async_context_manager_connects_and_disconnects(mock_ib) -> None:
    mock_ib.isConnected.side_effect = [False, True, True]  # initial False, then connected
    client = IBKRClient(role="cli", settings=Settings(), ib=mock_ib, backoff_seconds=())
    async with client as c:
        assert c is client
    mock_ib.connectAsync.assert_awaited_once()
    mock_ib.disconnect.assert_called_once()


async def test_connect_uses_configured_market_data_type(mock_ib) -> None:
    s = _settings_with_paper(paper=True)
    s.ibkr.market_data_type = 1  # live
    client = IBKRClient(role="cli", settings=s, ib=mock_ib, backoff_seconds=())
    mock_ib.isConnected.return_value = False
    await client.connect()
    mock_ib.reqMarketDataType.assert_called_once_with(1)
