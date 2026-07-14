from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from optionsbot.config import HermesWebhookSettings
from optionsbot.daemon.event_webhook import EventWebhookPublisher


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeClient:
    calls: list[tuple[str, bytes, dict[str, str]]] = []
    statuses: list[int] = [200]

    def __init__(self, **kwargs: Any) -> None:
        del kwargs

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        del args

    async def post(
        self, url: str, *, content: bytes, headers: dict[str, str]
    ) -> _FakeResponse:
        self.calls.append((url, content, headers))
        return _FakeResponse(self.statuses.pop(0))


def _settings(**overrides: Any) -> HermesWebhookSettings:
    values: dict[str, Any] = {
        "enabled": True,
        "secret": SecretStr("test-secret"),
        "retries": 0,
    }
    values.update(overrides)
    return HermesWebhookSettings(**values)


async def test_deliver_signs_exact_timestamp_dot_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeClient.calls = []
    _FakeClient.statuses = [200]
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    publisher = EventWebhookPublisher(_settings())

    assert await publisher.deliver("fill", "SPY filled", details={"order_id": 7})

    url, body, headers = _FakeClient.calls[0]
    assert url.endswith("/webhooks/optionsbot-events")
    payload = json.loads(body)
    assert payload["type"] == "fill"
    expected = hmac.new(
        b"test-secret",
        headers["X-Webhook-Timestamp"].encode() + b"." + body,
        hashlib.sha256,
    ).hexdigest()
    assert headers["X-Webhook-Signature-V2"] == expected
    assert headers["X-Request-ID"]


async def test_delivery_retries_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeClient.calls = []
    _FakeClient.statuses = [503, 200]
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    publisher = EventWebhookPublisher(_settings(retries=1))

    assert await publisher.deliver("daemon-down", "down", severity="critical")
    assert len(_FakeClient.calls) == 2


def test_enabled_delivery_requires_secret() -> None:
    with pytest.raises(ValidationError, match="secret is required"):
        HermesWebhookSettings(enabled=True)


@pytest.mark.parametrize(
    "url", ["https://127.0.0.1/webhook", "http://example.com/webhook"]
)
def test_delivery_rejects_non_loopback_http(url: str) -> None:
    with pytest.raises(ValidationError, match="loopback"):
        HermesWebhookSettings(url=url)
