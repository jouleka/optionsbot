"""Recent-news headlines via yfinance (IBK-108).

yfinance is the free, zero-config v1 source (same as the earnings path in
events.py). Its ``.news`` shape has shifted across versions -- recent versions
nest fields under a ``content`` key -- so parsing is defensive and ANY failure
yields an empty list (news is never load-bearing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import yfinance as yf  # type: ignore[import-untyped]
from sqlalchemy import Engine, delete, insert, select

from optionsbot.storage.schema import symbol_news

log = logging.getLogger(__name__)

_MAX_HEADLINES = 5
_RECENT_DAYS = 7


@dataclass(frozen=True, slots=True)
class Headline:
    title: str
    publisher: str
    published_ts: datetime | None
    link: str | None


def _parse_published(value: Any) -> datetime | None:
    """yfinance gives an epoch int (old) or an ISO string (new)."""
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=UTC)
        if isinstance(value, str):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    except (ValueError, OSError, OverflowError):
        return None
    return None


def _parse_item(item: dict[str, Any]) -> Headline | None:
    """Parse one yfinance news item (new ``content``-nested OR old flat shape)."""
    content = item.get("content", item)
    title = content.get("title")
    if not title:
        return None
    provider = content.get("provider")
    publisher = (
        (provider.get("displayName") if isinstance(provider, dict) else None)
        or item.get("publisher")
        or "unknown"
    )
    published = _parse_published(content.get("pubDate") or item.get("providerPublishTime"))
    canonical = content.get("canonicalUrl")
    link = (canonical.get("url") if isinstance(canonical, dict) else None) or item.get("link")
    return Headline(
        title=str(title), publisher=str(publisher), published_ts=published, link=link
    )


def recent_news(symbol: str) -> list[Headline]:
    """Up to 5 recent (<=7d) headlines for ``symbol`` via yfinance, newest first.

    Returns ``[]`` on ANY failure (network/parse) -- news is never load-bearing.
    """
    try:
        raw = yf.Ticker(symbol).news or []
    except Exception:  # noqa: BLE001 -- yfinance raises varied network/parse errors
        return []
    parsed: list[Headline] = []
    for item in raw:
        if isinstance(item, dict):
            h = _parse_item(item)
            if h is not None:
                parsed.append(h)
    cutoff = datetime.now(UTC) - timedelta(days=_RECENT_DAYS)
    fresh = [h for h in parsed if h.published_ts is None or h.published_ts >= cutoff]
    fresh.sort(
        key=lambda h: h.published_ts or datetime.min.replace(tzinfo=UTC), reverse=True
    )
    return fresh[:_MAX_HEADLINES]


def _headline_dict(h: Headline) -> dict[str, Any]:
    return {
        "title": h.title,
        "publisher": h.publisher,
        "published_ts": h.published_ts.isoformat() if h.published_ts else None,
        "link": h.link,
    }


def refresh_news_if_stale(symbol: str, engine: Engine, throttle_hours: int = 6) -> None:
    """Refresh ``symbol``'s cached headlines if missing or older than throttle_hours.

    Self-contained + graceful: NEVER raises (a yfinance/DB hiccup leaves the cache
    as-is). Called from scan_symbol so news refreshes at most every throttle_hours
    per symbol regardless of scan cadence.
    """
    try:
        now = datetime.now(UTC)
        with engine.connect() as conn:
            row = conn.execute(
                select(symbol_news.c.fetched_at).where(symbol_news.c.symbol == symbol)
            ).first()
        if row is not None and row.fetched_at is not None:
            fetched_at = row.fetched_at
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=UTC)
            if now - fetched_at < timedelta(hours=throttle_hours):
                return
        payload = [_headline_dict(h) for h in recent_news(symbol)]
        with engine.begin() as conn:
            conn.execute(delete(symbol_news).where(symbol_news.c.symbol == symbol))
            conn.execute(
                insert(symbol_news).values(
                    symbol=symbol, fetched_at=now, headlines_json=payload
                )
            )
    except Exception:  # noqa: BLE001 -- news is best-effort; never break the caller
        log.exception("refresh_news_if_stale failed for %s", symbol)
