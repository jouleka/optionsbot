"""Tests for entitled IBKR API news ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from optionsbot.config import Settings
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.news import NewsClient


@pytest.fixture()
def news_client(mock_ib: MagicMock) -> tuple[NewsClient, MagicMock]:
    mock_ib.isConnected.return_value = True
    mock_ib.reqNewsProvidersAsync = AsyncMock()
    mock_ib.reqHistoricalNewsAsync = AsyncMock()
    client = IBKRClient(
        role="cli",
        settings=Settings(),
        ib=mock_ib,
        backoff_seconds=(),
    )
    resolver = MagicMock()
    resolver.stock = AsyncMock(return_value=SimpleNamespace(conId=4815747))
    return NewsClient(client, resolver=resolver), mock_ib


async def test_headlines_use_only_entitled_providers_and_strip_metadata(
    news_client: tuple[NewsClient, MagicMock],
) -> None:
    client, ib = news_client
    ib.reqNewsProvidersAsync.return_value = [
        SimpleNamespace(code="DJ-N", name="Dow Jones"),
        SimpleNamespace(code="BRFG", name="Briefing.com"),
    ]
    ib.reqHistoricalNewsAsync.return_value = [
        SimpleNamespace(
            time=datetime(2026, 7, 27, 17, 44, tzinfo=UTC),
            providerCode="DJ-N",
            articleId="DJ-N$123",
            headline="{A:800015:L:en}Nvidia &amp; chips fall",
        )
    ]

    rows = await client.headlines("nvda", limit=10)

    ib.reqHistoricalNewsAsync.assert_awaited_once_with(
        4815747,
        "DJ-N+BRFG",
        "",
        "",
        10,
        [],
    )
    assert rows == [
        {
            "title": "Nvidia & chips fall",
            "publisher": "Dow Jones",
            "published_ts": "2026-07-27T17:44:00+00:00",
            "link": None,
            "source": "IBKR_API_NEWS",
            "provider_code": "DJ-N",
            "article_id": "DJ-N$123",
        }
    ]


async def test_provider_entitlements_are_cached_on_connection(
    news_client: tuple[NewsClient, MagicMock],
) -> None:
    client, ib = news_client
    ib.reqNewsProvidersAsync.return_value = [
        SimpleNamespace(code="DJNL", name="Dow Jones Newsletters")
    ]
    ib.reqHistoricalNewsAsync.return_value = []

    await client.headlines("SPY")
    await client.headlines("QQQ")

    ib.reqNewsProvidersAsync.assert_awaited_once()
    assert ib.reqHistoricalNewsAsync.await_count == 2


async def test_no_entitled_provider_returns_empty_without_history_request(
    news_client: tuple[NewsClient, MagicMock],
) -> None:
    client, ib = news_client
    ib.reqNewsProvidersAsync.return_value = []

    assert await client.headlines("SPY") == []
    ib.reqHistoricalNewsAsync.assert_not_awaited()


async def test_headline_limit_is_bounded(
    news_client: tuple[NewsClient, MagicMock],
) -> None:
    client, _ib = news_client

    with pytest.raises(ValueError, match="between 1 and 50"):
        await client.headlines("SPY", limit=51)
