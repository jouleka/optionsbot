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

    Important: only the methods that ARE actually async on the real
    ``ib_async.IB`` are wired up as ``AsyncMock``. The sync wrappers
    (e.g., ``reqSecDefOptParams``, ``positions``, ``accountSummary``,
    ``isConnected``, ``disconnect``, ``reqMarketDataType``) are plain
    ``MagicMock``. Code that calls the async sibling (e.g.
    ``reqSecDefOptParamsAsync``) must mock the ``*Async`` name, not the
    sync wrapper.
    """
    ib = MagicMock()
    # Async methods (return coroutines on the real IB):
    ib.connectAsync = AsyncMock()
    ib.qualifyContractsAsync = AsyncMock()
    ib.reqTickersAsync = AsyncMock()
    ib.reqSecDefOptParamsAsync = AsyncMock()
    ib.reqHistoricalDataAsync = AsyncMock()
    # Sync methods (return values directly on the real IB):
    ib.disconnect = MagicMock()
    ib.isConnected = MagicMock(return_value=False)
    ib.reqMarketDataType = MagicMock()
    ib.reqMktData = MagicMock()
    ib.positions = MagicMock()
    ib.accountSummary = MagicMock()
    return ib
