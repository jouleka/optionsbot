"""Shared fixtures for IBKR layer tests.

``mock_ib`` provides an ``ib_async.IB``-shaped AsyncMock so unit tests
don't need a live IB Gateway. Specific call-site behavior is configured
per test by setting ``.return_value`` / ``.side_effect`` on the mock.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture()
def mock_ib() -> MagicMock:
    """A MagicMock standing in for ``ib_async.IB`` with async methods on AsyncMock.

    Methods callers may stub: connectAsync, disconnect, isConnected,
    reqMarketDataType, qualifyContractsAsync, reqMktData, reqTickersAsync,
    reqSecDefOptParams, reqHistoricalDataAsync, positions, accountSummary.
    """
    ib = MagicMock()
    ib.connectAsync = AsyncMock()
    ib.disconnect = MagicMock()
    ib.isConnected = MagicMock(return_value=False)
    ib.reqMarketDataType = MagicMock()
    ib.qualifyContractsAsync = AsyncMock()
    ib.reqMktData = MagicMock()
    ib.reqTickersAsync = AsyncMock()
    ib.reqSecDefOptParams = AsyncMock()
    ib.reqHistoricalDataAsync = AsyncMock()
    ib.positions = MagicMock()
    ib.accountSummary = MagicMock()
    return ib
