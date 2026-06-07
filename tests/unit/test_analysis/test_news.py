"""Tests for recent-news headlines (IBK-108)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import insert, select

from optionsbot.analysis.news import Headline, recent_news, refresh_news_if_stale
from optionsbot.storage.db import create_engine_for_path
from optionsbot.storage.schema import symbol_news

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_recent_news_parses_new_and_old_shapes() -> None:
    recent_iso = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    recent_epoch = int((datetime.now(UTC) - timedelta(days=2)).timestamp())
    stale_iso = (datetime.now(UTC) - timedelta(days=20)).isoformat()
    payload = [
        {"content": {"title": "NVDA pops", "pubDate": recent_iso,
                     "provider": {"displayName": "Reuters"},
                     "canonicalUrl": {"url": "https://r/1"}}},
        {"title": "Old-shape news", "publisher": "Bloomberg",
         "providerPublishTime": recent_epoch, "link": "https://b/2"},
        {"content": {"title": "Stale", "pubDate": stale_iso}},
        {"foo": "bar"},   # no title -> skipped
        "not-a-dict",     # skipped
    ]
    with patch("optionsbot.analysis.news.yf") as mock_yf:
        mock_yf.Ticker.return_value = MagicMock(news=payload)
        out = recent_news("NVDA")
    titles = [h.title for h in out]
    assert "NVDA pops" in titles and "Old-shape news" in titles
    assert "Stale" not in titles          # >7d dropped
    assert out[0].title == "NVDA pops"    # newest first
    assert out[0].publisher == "Reuters"
    assert out[0].link == "https://r/1"


def test_recent_news_graceful_on_exception() -> None:
    with patch("optionsbot.analysis.news.yf") as mock_yf:
        mock_yf.Ticker.side_effect = Exception("network down")
        assert recent_news("AAPL") == []


def test_recent_news_caps_at_five() -> None:
    recent = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    payload = [{"content": {"title": f"H{i}", "pubDate": recent}} for i in range(8)]
    with patch("optionsbot.analysis.news.yf") as mock_yf:
        mock_yf.Ticker.return_value = MagicMock(news=payload)
        assert len(recent_news("X")) == 5


@pytest.fixture()
def news_engine(tmp_path: Path):  # type: ignore[no-untyped-def]
    from alembic.config import Config

    from alembic import command

    db_path = tmp_path / "news.db"
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")
    return create_engine_for_path(db_path)


def test_refresh_fetches_when_missing(news_engine) -> None:  # type: ignore[no-untyped-def]
    with patch(
        "optionsbot.analysis.news.recent_news",
        return_value=[Headline("T", "P", None, "L")],
    ) as mock_recent:
        refresh_news_if_stale("NVDA", news_engine)
        mock_recent.assert_called_once()
    with news_engine.connect() as conn:
        row = conn.execute(
            select(symbol_news).where(symbol_news.c.symbol == "NVDA")
        ).first()
    assert row is not None
    assert row.headlines_json[0]["title"] == "T"


def test_refresh_skips_when_fresh(news_engine) -> None:  # type: ignore[no-untyped-def]
    with news_engine.begin() as conn:
        conn.execute(insert(symbol_news).values(
            symbol="NVDA", fetched_at=datetime.now(UTC), headlines_json=[]))
    with patch("optionsbot.analysis.news.recent_news") as mock_recent:
        refresh_news_if_stale("NVDA", news_engine)
        mock_recent.assert_not_called()


def test_refresh_refetches_when_stale(news_engine) -> None:  # type: ignore[no-untyped-def]
    with news_engine.begin() as conn:
        conn.execute(insert(symbol_news).values(
            symbol="NVDA", fetched_at=datetime.now(UTC) - timedelta(hours=12),
            headlines_json=[]))
    with patch(
        "optionsbot.analysis.news.recent_news",
        return_value=[Headline("New", "P", None, None)],
    ) as mock_recent:
        refresh_news_if_stale("NVDA", news_engine, throttle_hours=6)
        mock_recent.assert_called_once()
    with news_engine.connect() as conn:
        row = conn.execute(
            select(symbol_news).where(symbol_news.c.symbol == "NVDA")
        ).first()
    assert row.headlines_json[0]["title"] == "New"


def test_refresh_graceful_when_recent_news_raises(news_engine) -> None:  # type: ignore[no-untyped-def]
    with patch("optionsbot.analysis.news.recent_news", side_effect=Exception("boom")):
        refresh_news_if_stale("NVDA", news_engine)  # must NOT raise
    with news_engine.connect() as conn:
        row = conn.execute(
            select(symbol_news).where(symbol_news.c.symbol == "NVDA")
        ).first()
    assert row is None  # nothing upserted on failure


def test_recent_news_handles_tz_naive_pubdate() -> None:
    # yfinance sometimes returns an ISO date with NO offset -> naive datetime.
    # It must not crash the recency comparison (which uses a tz-aware cutoff).
    naive_iso = (datetime.now(UTC) - timedelta(days=1)).replace(tzinfo=None).isoformat()
    payload = [{"content": {"title": "Naive date", "pubDate": naive_iso}}]
    with patch("optionsbot.analysis.news.yf") as mock_yf:
        mock_yf.Ticker.return_value = MagicMock(news=payload)
        out = recent_news("NVDA")
    assert [h.title for h in out] == ["Naive date"]  # kept, not crashed
