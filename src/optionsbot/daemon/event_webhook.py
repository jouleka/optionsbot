"""Signed zero-LLM operational events delivered to the local Hermes gateway."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from optionsbot.config import HermesWebhookSettings, load_settings

log = logging.getLogger(__name__)

ALLOWED_EVENT_TYPES = frozenset(
    {"fill", "stop-hit", "reconcile-mismatch", "daemon-down", "rth-acceptance"}
)


class EventWebhookPublisher:
    """Fire-and-forget publisher; failures never affect trading state."""

    def __init__(self, settings: HermesWebhookSettings) -> None:
        self._settings = settings
        self._tasks: set[asyncio.Task[bool]] = set()

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def emit(
        self,
        event_type: str,
        summary: str,
        *,
        severity: str = "info",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Schedule delivery without blocking the broker callback/tick."""
        if not self.enabled:
            return
        task = asyncio.create_task(
            self.deliver(event_type, summary, severity=severity, details=details)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def deliver(
        self,
        event_type: str,
        summary: str,
        *,
        severity: str = "info",
        details: dict[str, Any] | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        if event_type not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"unsupported event type: {event_type}")
        secret = self._settings.secret
        if secret is None:  # protected by config validation; defensive for mutation
            return False
        payload = {
            "type": event_type,
            "severity": severity,
            "summary": summary[:1000],
            "occurred_at": datetime.now(UTC).isoformat(),
            "source": "optionsbot",
            "details": details or {},
        }
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        timestamp = str(int(time.time()))
        signature = hmac.new(
            secret.get_secret_value().encode(),
            timestamp.encode() + b"." + body,
            hashlib.sha256,
        ).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Timestamp": timestamp,
            "X-Webhook-Signature-V2": signature,
            "X-Request-ID": str(uuid.uuid4()),
        }
        attempts = self._settings.retries + 1
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            for attempt in range(attempts):
                try:
                    response = await client.post(self._settings.url, content=body, headers=headers)
                    if 200 <= response.status_code < 300:
                        return True
                    retryable = response.status_code >= 500
                    log.warning(
                        "Hermes event delivery rejected: event=%s status=%d",
                        event_type,
                        response.status_code,
                    )
                except httpx.HTTPError as exc:
                    retryable = True
                    log.warning(
                        "Hermes event delivery failed: event=%s error=%s",
                        event_type,
                        type(exc).__name__,
                    )
                if not retryable or attempt == attempts - 1:
                    break
                await asyncio.sleep(0.25 * (2**attempt))
        return False

    async def flush(self) -> None:
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)


async def _send_cli(event_type: str, summary: str, severity: str) -> int:
    publisher = EventWebhookPublisher(load_settings().hermes_webhook)
    if not publisher.enabled:
        log.error("Hermes webhook delivery is disabled")
        return 2
    return 0 if await publisher.deliver(event_type, summary, severity=severity) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a signed OptionsBot operational event")
    parser.add_argument("event_type", choices=sorted(ALLOWED_EVENT_TYPES))
    parser.add_argument("--summary", required=True)
    parser.add_argument("--severity", default="critical")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_send_cli(args.event_type, args.summary, args.severity)))


if __name__ == "__main__":
    main()
