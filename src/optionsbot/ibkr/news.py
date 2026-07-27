"""Entitled IBKR API news headlines.

Only providers returned by ``reqNewsProviders`` are queried.  This keeps the
client entitlement-driven: it works with IBKR's included API feeds today and
automatically includes an API-specific paid feed later if the account enables
one, without hard-coding or assuming a subscription.
"""

from __future__ import annotations

import html
import re
from datetime import UTC, date, datetime
from typing import cast

from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver

_PROVIDER_CODE_RE = re.compile(r"^[A-Z0-9_-]{1,24}$")
_LEADING_METADATA_RE = re.compile(r"^(?:\{[^{}]{0,240}\})+")


def _clean_headline(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = _LEADING_METADATA_RE.sub("", value).strip()
    text = html.unescape(text)
    return " ".join(text.split())[:500]


def _published_iso(value: object) -> str | None:
    if isinstance(value, datetime):
        observed = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return observed.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC).isoformat()
    return None


class NewsClient:
    """Read symbol-specific headlines from the Gateway's current entitlements."""

    def __init__(
        self,
        client: IBKRClient,
        resolver: ContractResolver | None = None,
    ) -> None:
        self._client = client
        self._resolver = resolver or ContractResolver(client)

    async def _providers(self) -> tuple[tuple[str, str], ...]:
        ib = self._client.ib
        cached = getattr(ib, "_optionsbot_news_providers", None)
        if isinstance(cached, tuple):
            return cached
        providers = await ib.reqNewsProvidersAsync()
        normalized: list[tuple[str, str]] = []
        for provider in providers or []:
            code = str(getattr(provider, "code", "")).strip()
            name = str(getattr(provider, "name", "")).strip()
            if _PROVIDER_CODE_RE.fullmatch(code):
                normalized.append((code, name or code))
        result = tuple(normalized)
        ib._optionsbot_news_providers = result  # type: ignore[attr-defined]
        return result

    async def headlines(self, symbol: str, *, limit: int = 10) -> list[dict[str, object]]:
        """Return recent entitled headlines for one qualified stock contract."""
        if limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50")
        await self._client.ensure_connected()
        contract = await self._resolver.stock(symbol.upper().strip())
        providers = await self._providers()
        if not providers:
            return []
        provider_names = dict(providers)
        raw_rows = await self._client.ib.reqHistoricalNewsAsync(
            int(contract.conId),
            "+".join(code for code, _name in providers),
            "",
            "",
            limit,
            [],
        )
        rows = cast("list[object]", raw_rows or [])
        headlines: list[dict[str, object]] = []
        for row in rows:
            title = _clean_headline(getattr(row, "headline", None))
            if not title:
                continue
            provider_code = str(getattr(row, "providerCode", "")).strip()
            article_id = str(getattr(row, "articleId", "")).strip()
            headlines.append(
                {
                    "title": title,
                    "publisher": provider_names.get(provider_code, provider_code or "IBKR"),
                    "published_ts": _published_iso(getattr(row, "time", None)),
                    "link": None,
                    "source": "IBKR_API_NEWS",
                    "provider_code": provider_code or None,
                    "article_id": article_id or None,
                }
            )
        return headlines
