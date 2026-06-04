"""Telegram bot client: thin httpx wrapper around sendMessage (IBK-65)."""

from __future__ import annotations

from typing import Any, cast

import httpx

_API_BASE = "https://api.telegram.org"
_DEFAULT_TIMEOUT = 10.0


class TelegramClient:
    """Async client for the Telegram Bot API sendMessage endpoint.

    Owns one ``httpx.AsyncClient`` for the daemon's lifetime. ``aclose()``
    closes the underlying client; the daemon calls this in shutdown.
    """

    def __init__(
        self,
        bot_token: str | None,
        chat_id: str | None,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._client = client if client is not None else httpx.AsyncClient(timeout=timeout)

    async def send_message(self, text: str, parse_mode: str | None = "MarkdownV2") -> int:
        """Send a message to the configured chat. Returns Telegram message_id.

        ``parse_mode`` defaults to ``"MarkdownV2"``; pass ``None`` for plain text.
        """
        if not self._bot_token:
            raise RuntimeError("TelegramClient: bot_token is not configured")
        if not self._chat_id:
            raise RuntimeError("TelegramClient: chat_id is not configured")
        url = f"{_API_BASE}/bot{self._bot_token}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if parse_mode is not None:
            payload["parse_mode"] = parse_mode
        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        body = response.json()
        return cast(int, body["result"]["message_id"])

    async def get_updates(
        self, offset: int | None = None, timeout: int = 30
    ) -> list[dict[str, Any]]:
        """Long-poll getUpdates. Returns the raw ``result`` list (possibly empty)."""
        if not self._bot_token:
            raise RuntimeError("TelegramClient: bot_token is not configured")
        url = f"{_API_BASE}/bot{self._bot_token}/getUpdates"
        params: dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        # The per-request read timeout MUST exceed the long-poll timeout, or httpx
        # would cancel the request mid-poll. The client's default 10s is too short.
        response = await self._client.post(url, json=params, timeout=timeout + 10)
        response.raise_for_status()
        body = response.json()
        return cast("list[dict[str, Any]]", body.get("result", []))

    async def aclose(self) -> None:
        await self._client.aclose()
