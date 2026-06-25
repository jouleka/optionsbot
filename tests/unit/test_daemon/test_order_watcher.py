"""Tests for the 1-minute order watcher (IBK-126): TTL sweep + notifications."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from alembic.config import Config
from sqlalchemy import Engine, insert, update

from alembic import command
from optionsbot.config import Settings
from optionsbot.daemon import order_watcher as _order_watcher_module
from optionsbot.daemon.context import DaemonContext
from optionsbot.daemon.order_watcher import run_orders_tick
from optionsbot.daemon.telegram_client import TelegramClient
from optionsbot.execution.orders import get_order, record_fill
from optionsbot.execution.reconcile import ReconcileSummary
from optionsbot.ibkr import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.storage.db import create_engine_for_path
from optionsbot.storage.schema import orders

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

NOW = datetime(2026, 6, 10, 15, 30, tzinfo=UTC)


def _insert_order(
    engine: Engine,
    status: str,
    *,
    submitted_ts: datetime | None = None,
    terminal_ts: datetime | None = None,
    ib_order_id: int | None = 11,
    last_error: str | None = None,
) -> int:
    with engine.begin() as conn:
        result = conn.execute(
            insert(orders).values(
                intent="open", symbol="SPY", strategy="bull_put_spread",
                legs_json=[], quantity=1, status=status, staged_ts=NOW,
                submitted_ts=submitted_ts, terminal_ts=terminal_ts,
                ib_order_id=ib_order_id, reprice_count=0, last_error=last_error,
            )
        )
        pk = result.inserted_primary_key
        assert pk is not None
        order_id = int(pk[0])
        conn.execute(
            update(orders).where(orders.c.id == order_id)
            .values(order_ref=f"obot-{order_id}")
        )
    return order_id


def _wire_exec(daemon_context: DaemonContext) -> MagicMock:
    order_client = MagicMock()
    order_client.cancel = AsyncMock()
    daemon_context.order_client = order_client
    return order_client


async def test_noop_without_order_client(daemon_context: DaemonContext) -> None:
    summary = await run_orders_tick(daemon_context)
    assert summary.expired == 0 and summary.notified == 0


async def test_ttl_expiry_requests_cancel_tracker_confirms(
    daemon_context: DaemonContext,
) -> None:
    from sqlalchemy import update as sa_update

    order_client = _wire_exec(daemon_context)
    stale = _insert_order(
        daemon_context.engine, "submitted",
        submitted_ts=datetime.now(UTC) - timedelta(minutes=30),
    )
    fresh = _insert_order(
        daemon_context.engine, "submitted", submitted_ts=datetime.now(UTC),
    )

    async def confirm(ib_order_id: int) -> None:  # the tracker's job
        with daemon_context.engine.begin() as conn:
            conn.execute(sa_update(orders).where(orders.c.id == stale)
                         .values(status="cancelled", terminal_ts=datetime.now(UTC)))

    order_client.cancel = AsyncMock(side_effect=confirm)
    summary = await run_orders_tick(daemon_context)
    assert summary.expired == 1
    order_client.cancel.assert_awaited_once_with(11)
    record = get_order(daemon_context.engine, stale)
    assert record is not None
    assert record.status == "cancelled"  # tracker confirmed, not the sweep
    assert "TTL" in (record.last_error or "")
    assert get_order(daemon_context.engine, fresh).status == "submitted"  # type: ignore[union-attr]


async def test_ttl_unconfirmed_cancel_retries_next_tick(
    daemon_context: DaemonContext,
) -> None:
    # If the broker never confirms (no Cancelled event), the row stays WORKING
    # and the sweep simply requests the cancel again — it must never mark the
    # row terminal itself (a fill may still be racing the cancel).
    order_client = _wire_exec(daemon_context)
    stale = _insert_order(
        daemon_context.engine, "submitted",
        submitted_ts=datetime.now(UTC) - timedelta(minutes=30),
    )
    first = await run_orders_tick(daemon_context)
    second = await run_orders_tick(daemon_context)
    assert first.expired == 1 and second.expired == 1
    assert order_client.cancel.await_count == 2
    assert get_order(daemon_context.engine, stale).status == "submitted"  # type: ignore[union-attr]


async def test_ttl_registry_miss_warns_once_and_leaves_row(
    daemon_context: DaemonContext,
) -> None:
    order_client = _wire_exec(daemon_context)
    order_client.cancel = AsyncMock(side_effect=ValueError("unknown order id 11"))
    stale = _insert_order(
        daemon_context.engine, "submitted",
        submitted_ts=datetime.now(UTC) - timedelta(minutes=30),
    )
    first = await run_orders_tick(daemon_context)
    second = await run_orders_tick(daemon_context)
    assert first.expired == 0 and second.expired == 0
    # Row stays working (we could NOT cancel it at the broker — abandoning would lie).
    assert get_order(daemon_context.engine, stale).status == "submitted"  # type: ignore[union-attr]
    # Warned exactly once across both ticks.
    warn_texts = [
        c.args[0] for c in daemon_context.telegram.send_message.await_args_list
        if "restart" in c.args[0] or "manually" in c.args[0]
    ]
    assert len(warn_texts) == 1


async def test_notifies_new_terminals_once(daemon_context: DaemonContext) -> None:
    _wire_exec(daemon_context)
    daemon_context.orders_notified_through = NOW - timedelta(hours=1)
    filled = _insert_order(
        daemon_context.engine, "filled", terminal_ts=datetime.now(UTC),
    )
    record_fill(
        daemon_context.engine, filled, exec_id="e1", side="SELL", price=1.2,
        qty=1, ts=NOW,
    )
    _insert_order(
        daemon_context.engine, "rejected", terminal_ts=datetime.now(UTC),
        last_error="insufficient margin",
    )
    # Old terminal (before the notify watermark) must NOT re-notify.
    _insert_order(
        daemon_context.engine, "cancelled", terminal_ts=NOW - timedelta(hours=2),
    )
    first = await run_orders_tick(daemon_context)
    second = await run_orders_tick(daemon_context)
    assert first.notified == 2
    assert second.notified == 0
    sent = [c.args[0] for c in daemon_context.telegram.send_message.await_args_list]
    assert any("filled" in m.lower() for m in sent)
    assert any("insufficient margin" in m for m in sent)


def _make_context(tmp_path: Path | None = None) -> DaemonContext:
    """Build a minimal DaemonContext for tests that need one without the fixture."""
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())

    db_path = tmp_path / "test.db"
    cfg = Config(str(_PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")
    engine = create_engine_for_path(db_path)

    settings = Settings()
    settings.storage.db_path = db_path
    settings.telegram.bot_token = "test-token"
    settings.telegram.chat_id = "test-chat"
    settings.scan.auto_screen = False

    ibkr = MagicMock(spec=IBKRClient)
    ibkr.connect = AsyncMock()
    ibkr.ensure_connected = AsyncMock()
    ibkr.disconnect = AsyncMock()

    tg = MagicMock(spec=TelegramClient)
    tg.send_message = AsyncMock(return_value=12345)
    tg.aclose = AsyncMock()

    resolver = ContractResolver(ibkr)
    return DaemonContext(
        settings=settings,
        engine=engine,
        ibkr=ibkr,
        resolver=resolver,
        telegram=tg,
    )


async def test_reconcile_runs_on_fixed_cadence_with_no_open_orders(
    monkeypatch: Any,
) -> None:
    # Build a context whose ledger has NO open orders. The fixed-cadence
    # reconcile must STILL run (so the position-compare can catch orphans).
    context = _make_context()
    context.order_client = MagicMock()
    context.last_reconcile_ts = datetime.now(UTC) - timedelta(minutes=10)
    context.settings.execution.reconcile_minutes = 5

    called: list[bool] = []

    async def fake_reconcile(engine: Any, client: Any, **kwargs: Any) -> Any:
        called.append("positions_snapshot" in kwargs and kwargs["positions_snapshot"] is not None)
        return ReconcileSummary(0, 0, 0, 0, 0, 0)

    monkeypatch.setattr("optionsbot.execution.reconcile.reconcile", fake_reconcile)
    # Ensure the open-orders guard would have BLOCKED the old code path.
    monkeypatch.setattr(
        "optionsbot.execution.orders.open_orders", lambda engine: []
    )

    await _order_watcher_module.run_orders_tick(context)
    assert called == [True]  # reconcile ran AND was given a positions snapshot


async def test_net_liq_returns_usd_for_eur_account(
    daemon_context: DaemonContext,
) -> None:
    # IBK-122: the daily-loss kill compares USD realized PnL against this, so
    # _net_liq must return the USD-converted net-liq, not the raw EUR base.
    from decimal import Decimal
    from unittest.mock import patch

    from optionsbot.daemon.order_watcher import _net_liq
    from optionsbot.ibkr.types import AccountSummary

    fake_pos = MagicMock()
    fake_pos.get_account_summary = AsyncMock(
        return_value=AccountSummary(
            net_liquidation=Decimal("5000"), buying_power=None,
            available_funds=Decimal("5000"), currency="EUR",
            fx_to_usd=Decimal("1.25"),
        )
    )
    with patch("optionsbot.ibkr.positions.PositionsClient", return_value=fake_pos):
        result = await _net_liq(daemon_context)
    assert result == 6250.0  # 5000 EUR x 1.25 = USD 6250
