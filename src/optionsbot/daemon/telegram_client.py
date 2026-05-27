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

    async def send_message(self, text: str) -> int:
        """Send a Markdown message to the configured chat. Returns Telegram message_id."""
        if not self._bot_token:
            raise RuntimeError("TelegramClient: bot_token is not configured")
        if not self._chat_id:
            raise RuntimeError("TelegramClient: chat_id is not configured")
        url = f"{_API_BASE}/bot{self._bot_token}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True,
        }
        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        body = response.json()
        return cast(int, body["result"]["message_id"])

    async def aclose(self) -> None:
        await self._client.aclose()
