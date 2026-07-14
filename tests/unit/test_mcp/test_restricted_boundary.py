"""Negative tests for the Hermes least-privilege MCP boundary (IBK-137)."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from sqlalchemy import Engine, insert, select
from sqlalchemy.exc import OperationalError

from optionsbot.config import Settings
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.control_intents import consume_control_intents
from optionsbot.execution.state import load_state
from optionsbot.mcp_server.intent_queue import control_intents, create_intent_engine
from optionsbot.mcp_server.restricted_context import RestrictedServerContext
from optionsbot.mcp_server.server import build_server
from optionsbot.mcp_server.tools import nightwatch
from optionsbot.storage.db import create_readonly_engine_for_path
from optionsbot.storage.schema import watchlist
from tests.unit.test_mcp.conftest import FakeCtx, get_tools


@pytest.mark.asyncio
async def test_restricted_server_exposes_only_bounded_surface() -> None:
    server = build_server(restricted=True)
    names = {tool.name for tool in await server.list_tools()}
    assert names == {
        "analyze",
        "control_intent_status",
        "daily_brief",
        "halt",
        "health",
        "latest_snapshot",
        "list_watchlist",
        "pending_picks",
        "positions",
        "request_exit",
        "score_breakdown",
        "submit_entry_review",
        "track_record",
    }
    assert {"add_to_watchlist", "remove_from_watchlist", "set_view_override"}.isdisjoint(
        names
    )


def test_restricted_server_imports_no_broker_or_execution_modules() -> None:
    code = """
import sys
from optionsbot.mcp_server.server import build_server
build_server(restricted=True)
forbidden = sorted(
    name for name in sys.modules
    if name == 'optionsbot.ibkr' or name.startswith('optionsbot.ibkr.')
    or name == 'optionsbot.execution' or name.startswith('optionsbot.execution.')
)
if forbidden:
    raise SystemExit('forbidden imports: ' + ','.join(forbidden))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_primary_database_engine_is_physically_read_only(
    mcp_engine: Engine, mcp_settings: Settings
) -> None:
    with mcp_engine.begin() as conn:
        conn.execute(
            insert(watchlist).values(symbol="SPY", added_at=datetime.now(UTC))
        )
    readonly = create_readonly_engine_for_path(mcp_settings.storage.db_path)
    try:
        with readonly.connect() as conn:
            assert conn.execute(select(watchlist.c.symbol)).scalar_one() == "SPY"
        with pytest.raises(OperationalError):
            with readonly.begin() as conn:
                conn.execute(
                    insert(watchlist).values(
                        symbol="QQQ", added_at=datetime.now(UTC)
                    )
                )
    finally:
        readonly.dispose()


def test_restricted_context_contains_no_broker_or_messaging_secrets(
    mcp_engine: Engine, tmp_path: Path
) -> None:
    intent_engine = create_intent_engine(tmp_path / "intents.db")
    context = RestrictedServerContext(
        engine=mcp_engine,
        intent_engine=intent_engine,
        max_pick_age_minutes=20,
    )
    assert context.broker_access is False
    assert not hasattr(context, "ibkr")
    assert not hasattr(context.settings, "telegram")
    assert not hasattr(context.settings, "ibkr")


def test_halt_is_queued_then_monotonically_consumed_by_daemon(
    mcp_engine: Engine, tmp_path: Path
) -> None:
    intent_path = tmp_path / "intents.db"
    intent_engine = create_intent_engine(intent_path)
    context = RestrictedServerContext(
        engine=mcp_engine,
        intent_engine=intent_engine,
        max_pick_age_minutes=20,
    )
    halt = get_tools(nightwatch.register)["halt"]
    result = halt("boundary drill", "HALT_OPTIONSBOT", FakeCtx(context))
    assert result["ok"] is True
    assert result["killed"] == "pending_daemon_consumption"
    assert load_state(mcp_engine).killed is False

    daemon_context = cast("DaemonContext", SimpleNamespace(engine=mcp_engine))
    assert consume_control_intents(daemon_context, intent_path) == 1
    assert load_state(mcp_engine).killed is True
    assert load_state(mcp_engine).reason == "boundary drill"
    with intent_engine.connect() as conn:
        row = conn.execute(select(control_intents)).one()
    assert row.status == "processed"

    # Consumption is idempotent; a second pass cannot clear or duplicate it.
    assert consume_control_intents(daemon_context, intent_path) == 0
    assert load_state(mcp_engine).killed is True
