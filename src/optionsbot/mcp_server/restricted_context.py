"""Lifespan state for the least-privilege Hermes MCP endpoint."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from sqlalchemy import Engine

from optionsbot.mcp_server.intent_queue import create_intent_engine
from optionsbot.storage.db import create_readonly_engine_for_path

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


@dataclass(slots=True)
class RestrictedServerContext:
    """Only a read-only ledger plus the isolated control-intent queue."""

    engine: Engine
    intent_engine: Engine
    max_pick_age_minutes: int
    broker_access: bool = field(default=False, init=False)

    @property
    def settings(self) -> Any:
        """Minimal compatibility view; deliberately contains no credentials."""
        return SimpleNamespace(
            execution=SimpleNamespace(max_pick_age_minutes=self.max_pick_age_minutes)
        )

    async def shutdown(self) -> None:
        self.engine.dispose()
        self.intent_engine.dispose()


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"restricted MCP requires {name}")
    return Path(value)


@asynccontextmanager
async def restricted_app_lifespan(
    _server: FastMCP,
) -> AsyncIterator[RestrictedServerContext]:
    """Build a context without loading the application's secret-bearing Settings."""
    primary_path = _required_path("OPTIONSBOT_MCP_DB_PATH")
    intent_path = _required_path("OPTIONSBOT_MCP_INTENT_DB_PATH")
    max_age = int(os.environ.get("OPTIONSBOT_MCP_MAX_PICK_AGE_MINUTES", "20"))
    if not 1 <= max_age <= 60:
        raise RuntimeError("OPTIONSBOT_MCP_MAX_PICK_AGE_MINUTES must be within 1..60")
    ctx = RestrictedServerContext(
        engine=create_readonly_engine_for_path(primary_path),
        intent_engine=create_intent_engine(intent_path),
        max_pick_age_minutes=max_age,
    )
    try:
        yield ctx
    finally:
        await ctx.shutdown()
