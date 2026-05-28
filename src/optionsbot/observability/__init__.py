"""Observability primitives: structured logging, future tracing hooks."""

from optionsbot.observability.logging import (
    bind_log_context,
    configure_logging,
    get_logger,
)

__all__ = ["bind_log_context", "configure_logging", "get_logger"]
