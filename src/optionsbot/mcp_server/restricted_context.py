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
    zero_dte_only: bool = False
    zero_dte_entry_cutoff_minutes: int = 90
    opening_range_fvg_enabled: bool = False
    opening_range_minutes: int = 10
    opening_range_entry_window_minutes: int = 90
    broker_access: bool = field(default=False, init=False)

    @property
    def settings(self) -> Any:
        """Minimal compatibility view; deliberately contains no credentials."""
        return SimpleNamespace(
            execution=SimpleNamespace(
                max_pick_age_minutes=self.max_pick_age_minutes,
                zero_dte_only=self.zero_dte_only,
                zero_dte_entry_cutoff_minutes=self.zero_dte_entry_cutoff_minutes,
            ),
            scan=SimpleNamespace(
                opening_range_fvg_enabled=self.opening_range_fvg_enabled,
                opening_range_minutes=self.opening_range_minutes,
                opening_range_entry_window_minutes=(
                    self.opening_range_entry_window_minutes
                ),
            ),
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
    zero_dte_raw = os.environ.get("OPTIONSBOT_MCP_ZERO_DTE_ONLY", "false").lower()
    if zero_dte_raw not in {"true", "false"}:
        raise RuntimeError("OPTIONSBOT_MCP_ZERO_DTE_ONLY must be true or false")
    cutoff = int(
        os.environ.get("OPTIONSBOT_MCP_ZERO_DTE_ENTRY_CUTOFF_MINUTES", "90")
    )
    if not 30 <= cutoff <= 240:
        raise RuntimeError(
            "OPTIONSBOT_MCP_ZERO_DTE_ENTRY_CUTOFF_MINUTES must be within 30..240"
        )
    opening_range_raw = os.environ.get(
        "OPTIONSBOT_MCP_OPENING_RANGE_FVG_ENABLED", "false"
    ).lower()
    if opening_range_raw not in {"true", "false"}:
        raise RuntimeError(
            "OPTIONSBOT_MCP_OPENING_RANGE_FVG_ENABLED must be true or false"
        )
    opening_range_minutes = int(
        os.environ.get("OPTIONSBOT_MCP_OPENING_RANGE_MINUTES", "10")
    )
    opening_range_entry_window_minutes = int(
        os.environ.get("OPTIONSBOT_MCP_OPENING_RANGE_ENTRY_WINDOW_MINUTES", "90")
    )
    if not 5 <= opening_range_minutes <= 30:
        raise RuntimeError("OPTIONSBOT_MCP_OPENING_RANGE_MINUTES must be within 5..30")
    if not 30 <= opening_range_entry_window_minutes <= 390:
        raise RuntimeError(
            "OPTIONSBOT_MCP_OPENING_RANGE_ENTRY_WINDOW_MINUTES must be within 30..390"
        )
    if opening_range_entry_window_minutes <= opening_range_minutes:
        raise RuntimeError("opening-range entry window must end after the range")
    ctx = RestrictedServerContext(
        engine=create_readonly_engine_for_path(primary_path),
        intent_engine=create_intent_engine(intent_path),
        max_pick_age_minutes=max_age,
        zero_dte_only=zero_dte_raw == "true",
        zero_dte_entry_cutoff_minutes=cutoff,
        opening_range_fvg_enabled=opening_range_raw == "true",
        opening_range_minutes=opening_range_minutes,
        opening_range_entry_window_minutes=opening_range_entry_window_minutes,
    )
    try:
        yield ctx
    finally:
        await ctx.shutdown()
