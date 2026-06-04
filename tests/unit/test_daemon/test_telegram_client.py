"""Tests for TelegramClient (IBK-65)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from optionsbot.daemon.telegram_client import TelegramClient


def _mock_response(status: int = 200, json_data=None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.is_success = status < 400
    resp.json = MagicMock(return_value=json_data or {"ok": True, "result": {"message_id": 99}})
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err", request=MagicMock(), response=resp,
        )
    return resp


async def test_send_message_posts_to_telegram_api() -> None:
    client = TelegramClient(bot_token="abc:123", chat_id="9999")
    with patch.object(
        client._client, "post", new=AsyncMock(return_value=_mock_response()),
    ) as mock_post:
        msg_id = await client.send_message("hello world")
    mock_post.assert_awaited_once()
    args, kwargs = mock_post.await_args
    assert "abc:123" in args[0]
    assert "sendMessage" in args[0]
    payload = kwargs["json"]
    assert payload["chat_id"] == "9999"
    assert payload["text"] == "hello world"
    assert payload["parse_mode"] == "MarkdownV2"
    assert msg_id == 99
    await client.aclose()


async def test_send_message_returns_message_id_from_response() -> None:
    client = TelegramClient(bot_token="t", chat_id="c")
    with patch.object(
        client._client, "post",
        new=AsyncMock(
            return_value=_mock_response(
                json_data={"ok": True, "result": {"message_id": 4242}}
            )
        ),
    ):
        msg_id = await client.send_message("x")
    assert msg_id == 4242
    await client.aclose()


async def test_send_message_raises_runtimeerror_when_no_token() -> None:
    client = TelegramClient(bot_token=None, chat_id="c")
    with pytest.raises(RuntimeError, match="bot_token"):
        await client.send_message("x")
    await client.aclose()


async def test_send_message_raises_runtimeerror_when_no_chat_id() -> None:
    client = TelegramClient(bot_token="t", chat_id=None)
    with pytest.raises(RuntimeError, match="chat_id"):
        await client.send_message("x")
    await client.aclose()


async def test_send_message_propagates_http_error() -> None:
    client = TelegramClient(bot_token="t", chat_id="c")
    with patch.object(
        client._client, "post",
        new=AsyncMock(return_value=_mock_response(status=429)),
    ):
        with pytest.raises(httpx.HTTPStatusError):
            await client.send_message("x")
    await client.aclose()


async def test_aclose_closes_underlying_client() -> None:
    client = TelegramClient(bot_token="t", chat_id="c")
    with patch.object(client._client, "aclose", new=AsyncMock()) as mock_close:
        await client.aclose()
    mock_close.assert_awaited_once()


# --- IBK-102: get_updates + parse_mode ---

def _client_with_response(json_body: dict) -> tuple[TelegramClient, MagicMock]:
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.raise_for_status = MagicMock()
    http = MagicMock()
    http.post = AsyncMock(return_value=resp)
    return TelegramClient("tok", "123", client=http), http


async def test_get_updates_returns_result_list_and_passes_offset() -> None:
    tc, http = _client_with_response({"ok": True, "result": [{"update_id": 5}]})
    out = await tc.get_updates(offset=5, timeout=30)
    assert out == [{"update_id": 5}]
    args, kwargs = http.post.await_args
    assert kwargs["json"]["offset"] == 5
    assert kwargs["json"]["timeout"] == 30


async def test_send_message_plain_text_omits_parse_mode() -> None:
    tc, http = _client_with_response({"ok": True, "result": {"message_id": 9}})
    await tc.send_message("hi", parse_mode=None)
    assert "parse_mode" not in http.post.await_args.kwargs["json"]


async def test_send_message_defaults_to_markdownv2() -> None:
    tc, http = _client_with_response({"ok": True, "result": {"message_id": 9}})
    await tc.send_message("hi")
    assert http.post.await_args.kwargs["json"]["parse_mode"] == "MarkdownV2"
