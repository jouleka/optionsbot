"""Telegram bot client (httpx-based async wrapper). Filled in Task 4 / IBK-65."""

from __future__ import annotations


class TelegramClient:
    """Async client wrapping the Telegram Bot API ``sendMessage`` endpoint.

    Real implementation lands in Task 4. This stub exists so the daemon
    context + tests can reference the type during Task 1's scaffold.
    """

    def __init__(self, bot_token: str | None, chat_id: str | None) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id

    async def send_message(self, text: str) -> int:
        raise NotImplementedError("TelegramClient.send_message lands in IBK-65 (Task 4)")

    async def aclose(self) -> None:
        return None
