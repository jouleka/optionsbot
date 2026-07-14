"""Structured logging configuration via structlog.

Two output formats:

- ``env="dev"`` -> pretty console renderer (colored, human-readable),
  for interactive runs.
- ``env="prod"`` -> JSON renderer, one event per line, for systemd /
  journalctl / log shipping.

stdlib ``logging`` is configured to pipe through structlog's
``ProcessorFormatter`` so legacy ``logging.getLogger(__name__)`` calls
in the existing codebase still go through the same pipeline and
honor the same format.

``bind_log_context(**kwargs)`` is a context manager that binds key/value
pairs into the structlog contextvars for the duration of the block --
use this to attach a ``scan_run_id`` to every log line emitted during
one scheduler tick.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import structlog

_TELEGRAM_API_TOKEN_RE = re.compile(
    r"(?i)(https://api\.telegram\.org/bot)[^/\s?]+"
)
_TELEGRAM_BOT_TOKEN_RE = re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}\b")


def _redact_log_secrets(
    _logger: Any, _method_name: str, event_dict: structlog.typing.EventDict
) -> structlog.typing.EventDict:
    """Remove Telegram bot credentials from rendered log values.

    Both httpx request logs and ``HTTPStatusError`` strings include the full
    request URL. Telegram embeds the bot credential in that URL, so ordinary
    request/error logging would otherwise publish the credential to journald.
    """

    def redact(value: Any) -> Any:
        if isinstance(value, str):
            value = _TELEGRAM_API_TOKEN_RE.sub(r"\1<redacted>", value)
            return _TELEGRAM_BOT_TOKEN_RE.sub("<telegram-token-redacted>", value)
        if isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(redact(item) for item in value)
        return value

    for key, value in event_dict.items():
        event_dict[key] = redact(value)
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    env: Literal["dev", "prod"] = "dev",
) -> None:
    """Configure structlog + stdlib logging for the chosen environment.

    Idempotent: safe to call multiple times (e.g., from tests).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Shared processors that run before the final renderer. Order matters.
    # format_exc_info MUST be in the shared chain so log.exception() events
    # render the full traceback in BOTH dev console mode AND prod JSON mode --
    # without it, JSONRenderer emits {"exc_info": true} with no actual stack
    # trace, silently dropping the debugging signal callers expect.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.add_logger_name,
        structlog.processors.format_exc_info,
        _redact_log_secrets,
    ]

    if env == "prod":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Wire stdlib logging through structlog's ProcessorFormatter so
    # `logging.getLogger(__name__).info(...)` shares the format.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Replace existing handlers so re-configuration is idempotent and
    # doesn't double-emit on test re-runs.
    root.handlers = [handler]
    root.setLevel(level)


def get_logger(name: str | None = None) -> Any:
    """Return a structlog BoundLogger. Equivalent to structlog.get_logger."""
    return structlog.get_logger(name)


@contextmanager
def bind_log_context(**kwargs: Any) -> Iterator[None]:
    """Bind kwargs onto structlog contextvars for the duration of the block.

    Every log event emitted inside the block (including from stdlib
    logging via the ProcessorFormatter bridge) carries the bound fields.
    Nested binds stack; on exit only the keys bound IN this block are
    cleared, leaving outer-scope bindings untouched.
    """
    token_keys = list(kwargs.keys())
    structlog.contextvars.bind_contextvars(**kwargs)
    try:
        yield
    finally:
        structlog.contextvars.unbind_contextvars(*token_keys)
