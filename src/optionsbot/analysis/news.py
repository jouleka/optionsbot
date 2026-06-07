"""Recent-news headlines via yfinance (IBK-108).

yfinance is the free, zero-config v1 source (same as the earnings path in
events.py). Its ``.news`` shape has shifted across versions -- recent versions
nest fields under a ``content`` key -- so parsing is defensive and ANY failure
yields an empty list (news is never load-bearing).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import yfinance as yf  # type: ignore[import-untyped]

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
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
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
