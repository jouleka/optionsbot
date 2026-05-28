"""Tests for structured logging configuration."""

from __future__ import annotations

import json
import logging

import pytest
import structlog

from optionsbot.observability import bind_log_context, configure_logging


def test_configure_logging_is_idempotent() -> None:
    """Calling configure_logging multiple times shouldn't stack handlers."""
    configure_logging("INFO", env="dev")
    configure_logging("INFO", env="dev")
    root_handlers = logging.getLogger().handlers
    assert len(root_handlers) == 1, (
        f"Expected exactly 1 handler after idempotent re-config, got {len(root_handlers)}"
    )


def test_configure_logging_prod_env_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    """In prod env, log records render as one JSON object per line."""
    configure_logging("INFO", env="prod")
    log = structlog.get_logger("test")
    log.info("hello", foo="bar", n=42)
    captured = capsys.readouterr()
    line = captured.err.strip()
    assert line, "expected log output on stderr"
    parsed = json.loads(line)
    assert parsed["event"] == "hello"
    assert parsed["foo"] == "bar"
    assert parsed["n"] == 42
    assert parsed["level"] == "info"


def test_configure_logging_dev_env_emits_human_readable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """In dev env, output is the structlog console renderer (not JSON)."""
    configure_logging("INFO", env="dev")
    log = structlog.get_logger("test")
    log.info("hello-dev", widget="frob")
    captured = capsys.readouterr()
    out = captured.err
    assert "hello-dev" in out
    # Dev renderer doesn't emit valid JSON.
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.strip().split("\n")[-1])


def test_bind_log_context_attaches_field_to_emitted_event(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Inside bind_log_context(scan_run_id=...), log events carry that field."""
    configure_logging("INFO", env="prod")
    log = structlog.get_logger("test")
    with bind_log_context(scan_run_id="abc123"):
        log.info("inside")
    captured = capsys.readouterr()
    parsed = json.loads(captured.err.strip())
    assert parsed["scan_run_id"] == "abc123"


def test_bind_log_context_cleans_up_on_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """After the block exits, subsequent log events DO NOT carry the field."""
    configure_logging("INFO", env="prod")
    log = structlog.get_logger("test")
    with bind_log_context(scan_run_id="abc123"):
        pass  # bind + immediate unbind
    log.info("after")
    captured = capsys.readouterr()
    parsed = json.loads(captured.err.strip())
    assert "scan_run_id" not in parsed


def test_stdlib_logging_is_routed_through_structlog(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """logging.getLogger(__name__).info(...) emits via structlog format."""
    configure_logging("INFO", env="prod")
    stdlib_log = logging.getLogger("optionsbot.test")
    stdlib_log.info("from stdlib")
    captured = capsys.readouterr()
    line = captured.err.strip()
    parsed = json.loads(line)
    assert parsed["event"] == "from stdlib"
    assert parsed["logger"] == "optionsbot.test"
