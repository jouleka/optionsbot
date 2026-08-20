"""Send-only Telegram adapter; deliberately never calls getUpdates."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult


class OptionsBotTelegramOutboundAdapter(BasePlatformAdapter):
    supports_async_delivery = True

    def __init__(self, config: Any) -> None:
        super().__init__(config=config, platform=Platform.TELEGRAM)
        extra = getattr(config, "extra", {}) or {}
        self._token = os.getenv("OPTIONSBOT_TELEGRAM_BOT_TOKEN", "").strip()
        self._chat_id = str(extra.get("outbound_chat_id", "")).strip()

    @property
    def name(self) -> str:
        return "Telegram (OptionsBot outbound-only)"

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        del is_reconnect
        self._running = bool(self._token and self._chat_id)
        return self._running

    async def disconnect(self) -> None:
        self._running = False

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        del reply_to, metadata
        if not self._running:
            return SendResult(success=False, error="outbound adapter is not connected")
        if str(chat_id) != self._chat_id:
            return SendResult(success=False, error="delivery target is not allowlisted")
        text = content[:4000]
        try:
            message_id = await asyncio.to_thread(self._post_message, text)
        except Exception as exc:  # provider detail may contain sensitive URL; sanitize it
            return SendResult(
                success=False,
                error=f"Telegram send failed ({type(exc).__name__})",
                retryable=isinstance(exc, (TimeoutError, urllib.error.URLError)),
            )
        return SendResult(success=True, message_id=message_id)

    def _post_message(self, text: str) -> str:
        body = json.dumps({"chat_id": self._chat_id, "text": text}).encode()
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{self._token}/sendMessage",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            data = json.loads(response.read())
        if not data.get("ok"):
            raise RuntimeError("Telegram rejected message")
        return str(data["result"]["message_id"])

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {"name": "OptionsBot alerts", "type": "dm", "chat_id": str(chat_id)}


def _build_adapter(config: Any) -> OptionsBotTelegramOutboundAdapter:
    return OptionsBotTelegramOutboundAdapter(config)


def _is_connected(config: Any) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(
        os.getenv("OPTIONSBOT_TELEGRAM_BOT_TOKEN", "").strip()
        and str(extra.get("outbound_chat_id", "")).strip()
    )


def register(ctx: Any) -> None:
    # Register as "telegram" after the bundled adapter. The registry is
    # last-writer-wins, so Hermes uses this send-only implementation and no
    # second Telegram poller competes with OptionsBot's daemon.
    ctx.register_platform(
        name="telegram",
        label="Telegram (OptionsBot outbound-only)",
        adapter_factory=_build_adapter,
        check_fn=lambda: True,
        is_connected=_is_connected,
        required_env=["OPTIONSBOT_TELEGRAM_BOT_TOKEN"],
        max_message_length=4096,
        emoji="🔔",
    )
