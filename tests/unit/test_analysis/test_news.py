"""Tests for recent-news headlines (IBK-108)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from optionsbot.analysis.news import recent_news


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
