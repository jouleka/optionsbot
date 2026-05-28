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
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import structlog


def configure_logging(
    log_level: str = "INFO",
    env: Literal["dev", "prod"] = "dev",
) -> None:
    """Configure structlog + stdlib logging for the chosen environment.

    Idempotent: safe to call multiple times (e.g., from tests).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Shared processors that run before the final renderer. Order matters.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.add_logger_name,
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
