"""Narrow SQLite queue between the untrusted MCP process and trusted daemon.

The restricted MCP identity can read the trading ledger but cannot write it.
Its only writable state is this queue.  The daemon validates and translates
known intent types; arbitrary rows can never directly become broker orders.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    insert,
    select,
)

IntentKind = Literal["context_review"]

_ALLOWED_INTENT_KIND = "context_review"

metadata = MetaData()

control_intents = Table(
    "control_intents",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("intent_uid", String, nullable=False, unique=True),
    Column("kind", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("payload_json", JSON, nullable=False),
    Column("status", String, nullable=False, default="pending"),
    Column("processed_at", DateTime(timezone=True), nullable=True),
    Column("result_text", Text, nullable=True),
)


def create_intent_engine(path: Path | str) -> Engine:
    """Open/create the isolated intent queue and keep group access intact."""
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", future=True)

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    metadata.create_all(engine)
    if db_path.exists():
        try:
            # Group write access is the deliberate least-privilege boundary
            # between the restricted MCP process and the trusted daemon.
            os.chmod(db_path, 0o660)  # nosec B103
        except PermissionError:
            # The daemon normally opens a queue owned by the MCP identity via
            # their shared control group. Group members may use but not chmod
            # the file; deployment sets the mode/umask authoritatively.
            pass
    return engine


def enqueue_intent(
    engine: Engine,
    kind: IntentKind,
    payload: dict[str, Any],
    *,
    now: datetime | None = None,
) -> tuple[int, str]:
    """Append one typed intent and return its local id and opaque uid."""
    # Keep this as a runtime check as well as a type boundary.  The restricted
    # process is an untrusted writer, and callers written before the shadow-only
    # context boundary may still exist in an old deployment.
    if kind != _ALLOWED_INTENT_KIND:
        raise ValueError(
            "restricted action intent is disabled; only context_review is accepted"
        )
    created_at = now if now is not None else datetime.now(UTC)
    intent_uid = uuid.uuid4().hex
    with engine.begin() as conn:
        pk = conn.execute(
            insert(control_intents).values(
                intent_uid=intent_uid,
                kind=kind,
                created_at=created_at,
                payload_json=payload,
                status="pending",
            )
        ).inserted_primary_key
    assert pk is not None
    return int(pk[0]), intent_uid


def recent_intents(engine: Engine, limit: int = 20) -> list[dict[str, Any]]:
    """Return recent audit rows without exposing mutable database handles."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(control_intents)
            .order_by(control_intents.c.id.desc())
            .limit(max(1, min(int(limit), 100)))
        ).fetchall()
    return [dict(row._mapping) for row in rows]


def recent_proposal_decisions(
    engine: Engine, limit: int = 20
) -> list[dict[str, Any]]:
    """Return terminal Hermes proposal decisions for the next learning pass.

    The trusted daemon writes the authoritative result after rebuilding a
    proposal from live data.  Exposing this bounded, read-only projection lets
    Hermes adapt its next hypothesis to the exact gate that declined the prior
    one without granting it config, database, or broker mutation access.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                control_intents.c.id,
                control_intents.c.created_at,
                control_intents.c.processed_at,
                control_intents.c.payload_json,
                control_intents.c.status,
                control_intents.c.result_text,
            )
            .where(control_intents.c.kind == "entry_proposal")
            .where(control_intents.c.status.in_(("processed", "rejected")))
            .order_by(control_intents.c.id.desc())
            .limit(max(1, min(int(limit), 100)))
        ).fetchall()

    def _iso(value: datetime | None) -> str | None:
        if value is None:
            return None
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return aware.astimezone(UTC).isoformat()

    decisions: list[dict[str, Any]] = []
    for row in rows:
        payload = row.payload_json if isinstance(row.payload_json, dict) else {}
        decisions.append(
            {
                "intent_id": int(row.id),
                "symbol": payload.get("symbol"),
                "direction": payload.get("direction"),
                "iv_regime": payload.get("iv_regime"),
                "strategy": payload.get("strategy"),
                "confidence": payload.get("confidence"),
                "proposed_at": _iso(row.created_at),
                "processed_at": _iso(row.processed_at),
                "status": str(row.status),
                "decision": row.result_text,
            }
        )
    return decisions
