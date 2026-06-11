"""Order placement client (IBK-125) — the ONLY module touching ib_async's
order API.

Combos are built as guaranteed SMART-routed BAGs (US option-vs-option combos
execute atomically, any leg count — no client-side legging risk). Net-price
convention: BUY one unit of the bag as defined by the leg actions; a net
CREDIT structure therefore carries a NEGATIVE limit price. Single-leg
structures use the plain qualified Option contract with the leg's own action
and a positive premium.

Defense in depth: every MUTATION (place/modify/cancel) re-checks the
paper-only interlock from raw settings, independently of the execution
gate — even a bug above this layer cannot route an order to a live port
while ``execution.paper_only`` holds.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from optionsbot.config import PAPER_PORTS
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.ibkr.pacing import RateLimiter
from optionsbot.ibkr.types import (
    CommissionUpdate,
    ExecutionFill,
    MarginPreview,
    OptionRight,
    OrderStatusUpdate,
    PlacedOrder,
)

if TYPE_CHECKING:
    from ib_async import CommissionReport, Contract, Fill, Order, Trade

log = logging.getLogger(__name__)

# IBKR encodes "unset" doubles as DBL_MAX.
_UNSET_DOUBLE = 1.7976931348623157e308

_IB_SIDE = {"BOT": "BUY", "SLD": "SELL"}

LegMapping = Mapping[str, Any]


def _parse_ib_double(value: object) -> float | None:
    """IBKR margin/commission fields arrive as strings or DBL_MAX sentinels."""
    if value is None:
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if abs(parsed) >= _UNSET_DOUBLE:
        return None
    return parsed


class OrderClient:
    """Places/modifies/cancels combo orders and fans out order events.

    Mirrors the existing sub-client pattern: takes an ``IBKRClient`` (use
    role="exec" — order events are only delivered to the placing clientId)
    plus the shared ``ContractResolver`` for leg qualification.
    """

    def __init__(self, client: IBKRClient, resolver: ContractResolver) -> None:
        self._client = client
        self._resolver = resolver
        # Registry for modify/cancel: TWS API modifies by re-placing the SAME
        # order object (price/size/tif only).
        self._registry: dict[int, tuple[Contract, Order]] = {}
        # Order mutations are paced well under IBKR's 50 msg/s budget.
        self._rate = RateLimiter(max_calls=10, window_seconds=1.0)
        self._status_callbacks: list[Callable[[OrderStatusUpdate], None]] = []
        self._fill_callbacks: list[Callable[[ExecutionFill], None]] = []
        self._commission_callbacks: list[Callable[[CommissionUpdate], None]] = []
        self._subscribed = False

    # -- subscriptions ---------------------------------------------------------

    def on_status(self, callback: Callable[[OrderStatusUpdate], None]) -> None:
        self._status_callbacks.append(callback)

    def on_fill(self, callback: Callable[[ExecutionFill], None]) -> None:
        self._fill_callbacks.append(callback)

    def on_commission(self, callback: Callable[[CommissionUpdate], None]) -> None:
        self._commission_callbacks.append(callback)

    def _ensure_subscribed(self) -> None:
        if self._subscribed:
            return
        # IB-level events (not per-Trade) so re-bound orders after a reconnect
        # still reach the callbacks.
        self._client.ib.orderStatusEvent += self._handle_order_status
        self._client.ib.execDetailsEvent += self._handle_exec_details
        self._client.ib.commissionReportEvent += self._handle_commission
        self._subscribed = True

    # -- interlock ---------------------------------------------------------------

    def _assert_paper(self) -> None:
        settings = self._client.settings
        if not settings.execution.paper_only:
            return  # deliberate, documented live escape hatch (two config flips)
        if not settings.ibkr.paper:
            raise RuntimeError(
                "order mutation refused: paper-only interlock — ibkr.paper is false"
            )
        if settings.ibkr.port not in PAPER_PORTS:
            raise RuntimeError(
                "order mutation refused: paper-only interlock — port "
                f"{settings.ibkr.port} is not a recognized paper port (4002/7497)"
            )

    # -- building ---------------------------------------------------------------

    async def _build(
        self, symbol: str, legs: Sequence[LegMapping], limit_price: float
    ) -> tuple[Contract, str, float]:
        """legs_json-shaped dicts -> (contract, order action, order price).

        STK legs are dropped (Covered Call carries a synthetic existing-shares
        leg that must never be ordered). Multi-leg: guaranteed SMART BAG,
        action BUY, signed net price (negative = credit). Single leg: plain
        qualified Option, the leg's own action, positive premium.
        """
        from ib_async import ComboLeg, Contract

        option_legs = [leg for leg in legs if leg.get("sec_type", "OPT") == "OPT"]
        if not option_legs:
            raise ValueError(f"{symbol}: no option leg to order (got {len(legs)} legs)")
        specs: list[tuple[str, float, OptionRight]] = [
            (str(leg["expiry"]), float(leg["strike"]), leg["right"])
            for leg in option_legs
        ]
        qualified = await self._resolver.qualify_options(symbol, specs)
        missing = [spec for spec in specs if spec not in qualified]
        if missing:
            raise ValueError(f"{symbol}: could not qualify legs {missing}")

        if len(option_legs) == 1:
            leg = option_legs[0]
            action = str(leg["side"]).upper()
            return qualified[specs[0]], action, abs(limit_price)

        combo_legs = [
            ComboLeg(
                conId=qualified[spec].conId,
                ratio=int(leg.get("quantity", 1)),
                action=str(leg["side"]).upper(),
                exchange="SMART",
            )
            for leg, spec in zip(option_legs, specs, strict=True)
        ]
        bag = Contract(
            secType="BAG",
            symbol=symbol,
            currency="USD",
            exchange="SMART",  # guaranteed combo: legs execute atomically
            comboLegs=combo_legs,
        )
        return bag, "BUY", limit_price

    # -- order lifecycle -----------------------------------------------------------

    async def place_combo_limit(
        self,
        symbol: str,
        legs: Sequence[LegMapping],
        *,
        quantity: int,
        limit_price: float,
        order_ref: str,
        tif: str = "DAY",
    ) -> PlacedOrder:
        """Submit a limit order for the structure. ``limit_price`` is the
        signed net price per unit (negative = net credit)."""
        self._assert_paper()
        await self._client.ensure_connected()
        self._ensure_subscribed()
        contract, action, price = await self._build(symbol, legs, limit_price)

        from ib_async import LimitOrder

        order = LimitOrder(action, quantity, price, tif=tif, orderRef=order_ref)
        await self._rate.acquire()
        trade = self._client.ib.placeOrder(contract, order)
        ib_order_id = int(trade.order.orderId)
        self._registry[ib_order_id] = (contract, trade.order)
        log.info(
            "order placed: ref=%s id=%s %s %sx %s @ %s tif=%s",
            order_ref, ib_order_id, action, quantity, symbol, price, tif,
        )
        return PlacedOrder(
            ib_order_id=ib_order_id,
            order_ref=order_ref,
            action=action,
            limit_price=price,
            quantity=quantity,
        )

    async def whatif_combo(
        self,
        symbol: str,
        legs: Sequence[LegMapping],
        *,
        quantity: int,
        limit_price: float,
    ) -> MarginPreview:
        """Pre-trade margin/commission preview. Read-only — no interlock."""
        await self._client.ensure_connected()
        contract, action, price = await self._build(symbol, legs, limit_price)

        from ib_async import LimitOrder

        order = LimitOrder(action, quantity, price)
        result = await self._client.ib.whatIfOrderAsync(contract, order)
        # ib_async 2.1.0 returns a LIST of OrderState despite the annotation
        # (observed against a live paper Gateway); tolerate both shapes.
        state = (result[0] if result else None) if isinstance(result, list) else result
        if state is None:
            return MarginPreview(
                init_margin_change=None,
                maint_margin_change=None,
                equity_with_loan_change=None,
                commission=None,
                max_commission=None,
                warning="no whatIf response from IBKR",
            )
        warning = (state.warningText or "").strip() or None
        return MarginPreview(
            init_margin_change=_parse_ib_double(state.initMarginChange),
            maint_margin_change=_parse_ib_double(state.maintMarginChange),
            equity_with_loan_change=_parse_ib_double(state.equityWithLoanChange),
            commission=_parse_ib_double(state.commission),
            max_commission=_parse_ib_double(state.maxCommission),
            warning=warning,
        )

    async def modify_price(self, ib_order_id: int, *, new_limit_price: float) -> None:
        """Price-walk step: re-place the SAME order object with a new limit
        (the TWS API modify mechanism; price/size/tif changes only)."""
        self._assert_paper()
        entry = self._registry.get(ib_order_id)
        if entry is None:
            raise ValueError(f"unknown order id {ib_order_id} (not placed by this client)")
        contract, order = entry
        order.lmtPrice = new_limit_price
        await self._client.ensure_connected()
        await self._rate.acquire()
        self._client.ib.placeOrder(contract, order)
        log.info("order modified: id=%s new_limit=%s", ib_order_id, new_limit_price)

    async def cancel(self, ib_order_id: int) -> None:
        self._assert_paper()
        entry = self._registry.get(ib_order_id)
        if entry is None:
            raise ValueError(f"unknown order id {ib_order_id} (not placed by this client)")
        _, order = entry
        await self._client.ensure_connected()
        await self._rate.acquire()
        self._client.ib.cancelOrder(order)
        log.info("order cancel requested: id=%s", ib_order_id)

    async def open_order_refs(self) -> list[tuple[int, str | None, str]]:
        """(ib_order_id, orderRef, status) for every open order at IBKR.
        Minimal surface for now; IBK-128 reconciliation expands on it."""
        await self._client.ensure_connected()
        trades = await self._client.ib.reqAllOpenOrdersAsync()
        return [
            (
                int(trade.order.orderId),
                trade.order.orderRef or None,
                trade.orderStatus.status,
            )
            for trade in trades
        ]

    # -- event translation -----------------------------------------------------------

    def _handle_order_status(self, trade: Trade) -> None:
        update = OrderStatusUpdate(
            ib_order_id=int(trade.order.orderId),
            perm_id=int(trade.order.permId) or None,
            order_ref=trade.order.orderRef or None,
            status=trade.orderStatus.status,
            filled=float(trade.orderStatus.filled),
            remaining=float(trade.orderStatus.remaining),
            avg_fill_price=trade.orderStatus.avgFillPrice or None,
        )
        for callback in self._status_callbacks:
            callback(update)

    def _handle_exec_details(self, trade: Trade, fill: Fill) -> None:
        execution = fill.execution
        ts = execution.time if isinstance(execution.time, datetime) else datetime.now(UTC)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        record = ExecutionFill(
            ib_order_id=int(execution.orderId),
            order_ref=execution.orderRef or trade.order.orderRef or None,
            exec_id=execution.execId,
            side=_IB_SIDE.get(execution.side, execution.side),
            price=float(execution.price),
            qty=int(execution.shares),
            ts=ts,
            con_id=int(fill.contract.conId) or None,
            sec_type=fill.contract.secType,
        )
        for callback in self._fill_callbacks:
            callback(record)

    def _handle_commission(
        self, trade: Trade, fill: Fill | None, report: CommissionReport
    ) -> None:
        update = CommissionUpdate(
            exec_id=report.execId, commission=float(report.commission)
        )
        for callback in self._commission_callbacks:
            callback(update)
