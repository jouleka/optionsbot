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
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from optionsbot.config import PAPER_PORTS
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.ibkr.pacing import RateLimiter
from optionsbot.ibkr.types import (
    CommissionUpdate,
    ExecutionFill,
    MarginPreview,
    OpenOrderSnapshot,
    OptionRight,
    OrderStatusUpdate,
    PlacedOrder,
    ledger_row_id_from_ref,
)

if TYPE_CHECKING:
    from ib_async import CommissionReport, Contract, Fill, Order, Trade

log = logging.getLogger(__name__)

# IBKR encodes "unset" doubles as DBL_MAX.
_UNSET_DOUBLE = 1.7976931348623157e308

_IB_SIDE = {"BOT": "BUY", "SLD": "SELL"}
_KNOWN_FOREIGN_EXECUTION_TYPES = frozenset(
    {"STK", "FUT", "FOP", "CASH", "CFD", "BOND", "CMDTY", "FUND"}
)

LegMapping = Mapping[str, Any]
LegContract = tuple[int, int, str]
BrokerCallbackKind = Literal["orderStatus", "execDetails", "commissionReport"]
BrokerCallbackErrorHandler = Callable[[BrokerCallbackKind, Exception], None]
_CallbackPayload = TypeVar("_CallbackPayload")


def _parse_ib_double(value: object) -> float | None:
    """IBKR margin/commission fields arrive as strings or DBL_MAX sentinels."""
    if value is None:
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or abs(parsed) >= _UNSET_DOUBLE:
        return None
    return parsed


def _empty_commission_placeholder(report: Any) -> bool:
    """True only for ib_async's exact default reqExecutions placeholder."""
    return (
        report.execId == ""
        and isinstance(report.commission, (int, float))
        and not isinstance(report.commission, bool)
        and report.commission == 0
        and report.currency == ""
        and isinstance(report.realizedPNL, (int, float))
        and not isinstance(report.realizedPNL, bool)
        and report.realizedPNL == 0
        and isinstance(report.yield_, (int, float))
        and not isinstance(report.yield_, bool)
        and report.yield_ == 0
        and type(report.yieldRedemptionDate) is int
        and report.yieldRedemptionDate == 0
    )


def _validated_open_trade(trade: Any) -> tuple[int, str | None, str]:
    try:
        raw_order_id = trade.order.orderId
        raw_ref = trade.order.orderRef
        raw_status = trade.orderStatus.status
    except AttributeError as exc:
        raise ValueError("open-order row is malformed") from exc
    if type(raw_order_id) is not int or raw_order_id < 0:
        raise ValueError("open-order identity must be an exact nonnegative integer")
    if raw_ref is not None and not isinstance(raw_ref, str):
        raise ValueError("open-order reference is malformed")
    ref = raw_ref or None
    if not isinstance(raw_status, str) or not raw_status.strip():
        raise ValueError("open-order status is malformed")
    return raw_order_id, ref, raw_status


def _validate_bot_contract_order(contract: Any, order: Any) -> None:
    sec_type = contract.secType
    if sec_type not in {"OPT", "BAG"}:
        raise ValueError(f"bot open order has unknown security type {sec_type!r}")
    if not isinstance(contract.currency, str) or not contract.currency:
        raise ValueError("bot open-order currency is malformed")
    if contract.currency != contract.currency.upper():
        raise ValueError("bot open-order currency is malformed")
    if not isinstance(contract.symbol, str) or not contract.symbol.strip():
        raise ValueError("bot open-order symbol is malformed")
    if sec_type == "OPT":
        if (
            type(contract.conId) is not int
            or contract.conId <= 0
            or not isinstance(contract.multiplier, str)
            or not contract.multiplier.isascii()
            or not contract.multiplier.isdecimal()
            or int(contract.multiplier) <= 0
            or not isinstance(contract.lastTradeDateOrContractMonth, str)
            or len(contract.lastTradeDateOrContractMonth) != 8
            or not contract.lastTradeDateOrContractMonth.isascii()
            or not contract.lastTradeDateOrContractMonth.isdecimal()
            or not isinstance(contract.strike, (int, float))
            or isinstance(contract.strike, bool)
            or not math.isfinite(float(contract.strike))
            or contract.strike <= 0
            or contract.right not in {"C", "P"}
        ):
            raise ValueError("bot option open-order contract is malformed")
    else:
        combo_legs = contract.comboLegs
        if not isinstance(combo_legs, (list, tuple)) or not combo_legs:
            raise ValueError("bot combo open-order legs are malformed")
        seen: set[int] = set()
        for leg in combo_legs:
            if (
                type(leg.conId) is not int
                or leg.conId <= 0
                or leg.conId in seen
                or type(leg.ratio) is not int
                or leg.ratio <= 0
                or leg.action not in {"BUY", "SELL"}
                or leg.exchange != "SMART"
            ):
                raise ValueError("bot combo open-order leg is malformed")
            seen.add(leg.conId)
    raw_quantity = order.totalQuantity
    raw_limit = order.lmtPrice
    if (
        not isinstance(raw_quantity, (int, float))
        or isinstance(raw_quantity, bool)
        or not math.isfinite(float(raw_quantity))
        or raw_quantity <= 0
        or not float(raw_quantity).is_integer()
        or not isinstance(raw_limit, (int, float))
        or isinstance(raw_limit, bool)
        or not math.isfinite(float(raw_limit))
        or order.orderType != "LMT"
        or order.tif != "DAY"
        or order.action not in {"BUY", "SELL"}
    ):
        raise ValueError("bot open-order terms are malformed")


def _validate_adopted_bot_trade(trade: Any) -> None:
    try:
        contract = trade.contract
        order = trade.order
    except AttributeError as exc:
        raise ValueError("bot open-order row is malformed") from exc
    _validate_bot_contract_order(contract, order)


def _snapshot_bot_order(contract: Any, order: Any, *, status: str) -> OpenOrderSnapshot:
    raw_order_id = order.orderId
    raw_ref = order.orderRef
    if type(raw_order_id) is not int or raw_order_id <= 0:
        raise ValueError("bot open-order identity must be an exact positive integer")
    if ledger_row_id_from_ref(raw_ref) is None:
        raise ValueError("bot open-order reference is not canonical")
    if not isinstance(status, str) or not status.strip():
        raise ValueError("bot open-order status is malformed")
    _validate_bot_contract_order(contract, order)
    sec_type = contract.secType
    combo_legs: tuple[tuple[int, int, str, str], ...] = ()
    contract_con_id: int | None = None
    multiplier: int | None = None
    expiry: str | None = None
    strike: float | None = None
    right: OptionRight | None = None
    if sec_type == "BAG":
        combo_legs = tuple(
            (leg.conId, leg.ratio, leg.action, leg.exchange)
            for leg in contract.comboLegs
        )
    else:
        contract_con_id = contract.conId
        multiplier = int(contract.multiplier)
        expiry = contract.lastTradeDateOrContractMonth
        strike = float(contract.strike)
        right = contract.right
    return OpenOrderSnapshot(
        ib_order_id=raw_order_id,
        order_ref=raw_ref,
        status=status,
        sec_type=sec_type,
        symbol=contract.symbol,
        currency=contract.currency,
        exchange=contract.exchange,
        contract_con_id=contract_con_id,
        multiplier=multiplier,
        expiry=expiry,
        strike=strike,
        right=right,
        combo_legs=combo_legs,
        order_action=order.action,
        total_quantity=int(order.totalQuantity),
        order_type=order.orderType,
        tif=order.tif,
        limit_price=float(order.lmtPrice),
    )


def _snapshot_open_trade(trade: Any) -> OpenOrderSnapshot:
    order_id, ref, status = _validated_open_trade(trade)
    if ledger_row_id_from_ref(ref) is None:
        return OpenOrderSnapshot(order_id, ref, status)
    return _snapshot_bot_order(trade.contract, trade.order, status=status)


def _mutation_authority(snapshot: OpenOrderSnapshot) -> tuple[Any, ...]:
    """Immutable broker-mutation terms; status is evidence, not authority."""
    return (
        snapshot.ib_order_id,
        snapshot.order_ref,
        snapshot.sec_type,
        snapshot.symbol,
        snapshot.currency,
        snapshot.exchange,
        snapshot.contract_con_id,
        snapshot.multiplier,
        snapshot.expiry,
        snapshot.strike,
        snapshot.right,
        snapshot.combo_legs,
        snapshot.order_action,
        snapshot.total_quantity,
        snapshot.order_type,
        snapshot.tif,
        snapshot.limit_price,
    )


@dataclass(frozen=True, slots=True)
class _RegisteredOrder:
    contract: Contract
    order: Order
    authority: tuple[Any, ...]


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
        self._registry: dict[int, _RegisteredOrder] = {}
        self._pending_adoptions: dict[int, tuple[OpenOrderSnapshot, Any]] = {}
        # Order mutations are paced well under IBKR's 50 msg/s budget.
        self._rate = RateLimiter(max_calls=10, window_seconds=1.0)
        self._status_callbacks: list[Callable[[OrderStatusUpdate], None]] = []
        self._fill_callbacks: list[Callable[[ExecutionFill], None]] = []
        self._commission_callbacks: list[Callable[[CommissionUpdate], None]] = []
        self._callback_error_handler: BrokerCallbackErrorHandler | None = None
        self._subscribed = False

    # -- subscriptions ---------------------------------------------------------

    def on_status(self, callback: Callable[[OrderStatusUpdate], None]) -> None:
        self._status_callbacks.append(callback)

    def on_fill(self, callback: Callable[[ExecutionFill], None]) -> None:
        self._fill_callbacks.append(callback)

    def on_commission(self, callback: Callable[[CommissionUpdate], None]) -> None:
        self._commission_callbacks.append(callback)

    def on_callback_error(self, callback: BrokerCallbackErrorHandler) -> None:
        """Install the one fail-closed boundary for raw broker callbacks."""
        if self._subscribed:
            raise RuntimeError("callback error handler must be installed before subscription")
        if self._callback_error_handler is not None:
            raise RuntimeError("callback error handler is already installed")
        self._callback_error_handler = callback

    def _ensure_subscribed(self) -> None:
        if self._subscribed:
            return
        if self._callback_error_handler is None:
            raise RuntimeError("broker callback error handler is not installed")
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
    ) -> tuple[Contract, str, float, tuple[LegContract, ...]]:
        """legs_json-shaped dicts -> (contract, order action, order price).

        STK legs are dropped (Covered Call carries a synthetic existing-shares
        leg that must never be ordered). Multi-leg: guaranteed SMART BAG,
        action BUY, signed net price (negative = credit). Single leg: plain
        qualified Option, the leg's own action, positive premium.
        """
        from ib_async import ComboLeg, Contract

        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("order symbol must be nonblank")
        if (
            not isinstance(limit_price, (int, float))
            or isinstance(limit_price, bool)
            or not math.isfinite(float(limit_price))
        ):
            raise ValueError("order limit price must be finite")
        option_legs: list[LegMapping] = []
        specs: list[tuple[str, float, OptionRight]] = []
        leg_sides: list[str] = []
        leg_ratios: list[int] = []
        for leg in legs:
            sec_type = leg.get("sec_type", "OPT")
            if sec_type == "STK":
                continue
            if sec_type != "OPT":
                raise ValueError(f"unsupported or malformed security type {sec_type!r}")
            raw_expiry = leg.get("expiry")
            raw_strike = leg.get("strike")
            raw_right = leg.get("right")
            raw_side = leg.get("side")
            raw_ratio = leg.get("quantity", 1)
            leg_symbol = leg.get("symbol")
            if (
                not isinstance(raw_expiry, str)
                or len(raw_expiry) != 8
                or not raw_expiry.isdigit()
                or not isinstance(raw_strike, (int, float))
                or isinstance(raw_strike, bool)
                or not math.isfinite(float(raw_strike))
                or raw_strike <= 0
                or raw_right not in {"C", "P"}
                or raw_side not in {"buy", "sell"}
                or type(raw_ratio) is not int
                or raw_ratio <= 0
                or leg_symbol != symbol
            ):
                raise ValueError(f"{symbol}: malformed option leg evidence")
            spec = (raw_expiry, float(raw_strike), raw_right)
            if spec in specs:
                raise ValueError(f"{symbol}: duplicate option contract specification")
            option_legs.append(leg)
            specs.append(spec)
            leg_sides.append(raw_side.upper())
            leg_ratios.append(raw_ratio)
        if not option_legs:
            raise ValueError(f"{symbol}: no option leg to order (got {len(legs)} legs)")
        qualified = await self._resolver.qualify_options(symbol, specs)
        missing = [spec for spec in specs if spec not in qualified]
        if missing:
            raise ValueError(f"{symbol}: could not qualify legs {missing}")

        validated_contracts: list[Contract] = []
        leg_contracts_list: list[LegContract] = []
        currencies: set[str] = set()
        qualified_con_ids: set[int] = set()
        for spec in specs:
            contract = qualified[spec]
            expiry, strike, right = spec
            raw_con_id = contract.conId
            raw_multiplier = contract.multiplier
            raw_currency = contract.currency
            if type(raw_con_id) is not int or raw_con_id <= 0:
                raise ValueError(f"{symbol}: qualified contract ID is malformed")
            if raw_con_id in qualified_con_ids:
                raise ValueError(f"{symbol}: duplicate qualified contract identity")
            qualified_con_ids.add(raw_con_id)
            if (
                isinstance(raw_multiplier, str)
                and raw_multiplier
                and raw_multiplier.isdigit()
            ):
                multiplier = int(raw_multiplier)
            else:
                raise ValueError(f"{symbol}: qualified multiplier is malformed")
            if multiplier != 100:
                raise ValueError(f"{symbol}: qualified multiplier is unsupported")
            if (
                contract.secType != "OPT"
                or contract.symbol != symbol
                or contract.lastTradeDateOrContractMonth != expiry
                or not isinstance(contract.strike, (int, float))
                or isinstance(contract.strike, bool)
                or not math.isfinite(float(contract.strike))
                or float(contract.strike) != strike
                or contract.right != right
                or raw_currency != "USD"
            ):
                raise ValueError(f"{symbol}: qualified contract terms are malformed")
            currencies.add(raw_currency)
            validated_contracts.append(contract)
            leg_contracts_list.append((raw_con_id, multiplier, raw_currency))
        if len(currencies) != 1:
            raise ValueError(f"{symbol}: qualified legs have inconsistent currencies")
        currency = next(iter(currencies))
        leg_contracts = tuple(leg_contracts_list)

        if len(option_legs) == 1:
            return validated_contracts[0], leg_sides[0], abs(limit_price), leg_contracts

        combo_legs = [
            ComboLeg(
                conId=contract.conId,
                ratio=ratio,
                action=side,
                exchange="SMART",
            )
            for contract, ratio, side in zip(
                validated_contracts, leg_ratios, leg_sides, strict=True
            )
        ]
        bag = Contract(
            secType="BAG",
            symbol=symbol,
            currency=currency,
            exchange="SMART",  # guaranteed combo: legs execute atomically
            comboLegs=combo_legs,
        )
        return bag, "BUY", limit_price, leg_contracts

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
        is_closing: bool = False,
    ) -> PlacedOrder:
        """Submit a limit order for the structure. ``limit_price`` is the
        signed net price per unit (negative = net credit)."""
        if type(quantity) is not int or quantity <= 0:
            raise ValueError("order quantity must be a positive exact integer")
        if ledger_row_id_from_ref(order_ref) is None:
            raise ValueError("order reference must be canonical obot-<positive-int>")
        self._assert_paper()
        await self._client.ensure_connected()
        self._ensure_subscribed()
        contract, action, price, leg_contracts = await self._build(
            symbol, legs, limit_price
        )

        from ib_async import LimitOrder

        option_legs = [leg for leg in legs if leg.get("sec_type", "OPT") == "OPT"]
        broker_quantity = (
            quantity * int(option_legs[0].get("quantity", 1))
            if len(option_legs) == 1
            else quantity
        )
        order = LimitOrder(action, broker_quantity, price, tif=tif, orderRef=order_ref)
        if is_closing and len(option_legs) > 1 and math.isclose(
            float(price), 0.0, abs_tol=1e-9
        ):
            # IBKR rejects a zero-cost combo as "riskless" unless the API
            # transmits the exact advanced-error acknowledgement it returns.
            # Scope this narrowly to atomic CLOSES; entries never bypass the
            # warning and non-zero closes do not need it.
            order.advancedErrorOverride = "8229=COMBOPAYOUT"
        await self._rate.acquire()
        trade = self._client.ib.placeOrder(contract, order)
        raw_ib_order_id = trade.order.orderId
        if type(raw_ib_order_id) is not int or raw_ib_order_id <= 0:
            raise RuntimeError("broker acknowledgement has malformed order identity")
        ib_order_id = raw_ib_order_id
        registered_snapshot = _snapshot_bot_order(
            contract,
            trade.order,
            status="Registered",
        )
        self._registry[ib_order_id] = _RegisteredOrder(
            contract,
            trade.order,
            _mutation_authority(registered_snapshot),
        )
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
            leg_contracts=leg_contracts,
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
        if type(quantity) is not int or quantity <= 0:
            raise ValueError("what-if quantity must be a positive exact integer")
        await self._client.ensure_connected()
        contract, action, price, _ = await self._build(symbol, legs, limit_price)

        from ib_async import LimitOrder

        option_legs = [leg for leg in legs if leg.get("sec_type", "OPT") == "OPT"]
        broker_quantity = (
            quantity * int(option_legs[0].get("quantity", 1))
            if len(option_legs) == 1
            else quantity
        )
        order = LimitOrder(action, broker_quantity, price)
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
        if type(ib_order_id) is not int or ib_order_id <= 0:
            raise ValueError("modify order identity must be an exact positive integer")
        if (
            not isinstance(new_limit_price, (int, float))
            or isinstance(new_limit_price, bool)
            or not math.isfinite(float(new_limit_price))
        ):
            raise ValueError("modify limit price must be finite")
        self._assert_paper()
        if ib_order_id not in self._registry:
            raise ValueError(f"unknown order id {ib_order_id} (not placed by this client)")
        await self._client.ensure_connected()
        await self._rate.acquire()
        self._assert_paper()
        entry = self._registered_order(ib_order_id)
        contract = entry.contract
        order = entry.order
        # Same normalization as placement: only BAG combos use signed net
        # limits; a single-leg option's premium is always positive (a signed
        # walk target sent raw produced lmtPrice=-6.14 → IBKR Error 201).
        order.lmtPrice = new_limit_price if contract.secType == "BAG" else abs(new_limit_price)
        changed = _snapshot_bot_order(contract, order, status="Registered")
        # Mutating the existing Order object is IBKR's modify mechanism.  Make
        # that deliberate local mutation authoritative *before* placeOrder:
        # placeOrder can raise after the request crossed the socket, and an
        # emergency cancel must still be allowed to use the exact same order.
        # Previously the registry retained the pre-modify price on that path,
        # so cancel() rejected our own authorized change as "authority drift".
        self._registry[ib_order_id] = _RegisteredOrder(
            contract,
            order,
            _mutation_authority(changed),
        )
        self._client.ib.placeOrder(contract, order)
        # Fail closed if the synchronous broker call changed any protected
        # identity/contract/order term behind our back.  Normal IBKR updates
        # (permId/status) are intentionally outside the authority tuple.
        confirmed = _snapshot_bot_order(contract, order, status="Registered")
        if _mutation_authority(confirmed) != self._registry[ib_order_id].authority:
            raise RuntimeError(
                f"order id {ib_order_id} broker acknowledgement altered mutation authority"
            )
        log.info("order modified: id=%s new_limit=%s", ib_order_id, new_limit_price)

    async def cancel(self, ib_order_id: int) -> None:
        if type(ib_order_id) is not int or ib_order_id <= 0:
            raise ValueError("cancel order identity must be an exact positive integer")
        self._assert_paper()
        if ib_order_id not in self._registry:
            raise ValueError(f"unknown order id {ib_order_id} (not placed by this client)")
        await self._client.ensure_connected()
        await self._rate.acquire()
        self._assert_paper()
        entry = self._registered_order(ib_order_id)
        self._client.ib.cancelOrder(entry.order)
        log.info("order cancel requested: id=%s", ib_order_id)

    def _registered_order(self, ib_order_id: int) -> _RegisteredOrder:
        entry = self._registry.get(ib_order_id)
        if entry is None:
            raise ValueError(f"unknown order id {ib_order_id} (not placed by this client)")
        current = _snapshot_bot_order(
            entry.contract,
            entry.order,
            status="Registered",
        )
        if _mutation_authority(current) != entry.authority:
            raise ValueError(
                f"order id {ib_order_id} broker mutation authority drifted"
            )
        return entry

    def authorize_adoptions(self, snapshots: tuple[OpenOrderSnapshot, ...]) -> None:
        """Atomically authorize the exact snapshots validated by reconciliation."""
        approved: dict[int, _RegisteredOrder] = {}
        seen: set[int] = set()
        for snapshot in snapshots:
            if snapshot.ib_order_id in seen:
                raise ValueError("adoption authorization contains duplicate order identity")
            seen.add(snapshot.ib_order_id)
            pending = self._pending_adoptions.get(snapshot.ib_order_id)
            if pending is None or pending[0] != snapshot:
                raise ValueError(
                    f"order id {snapshot.ib_order_id} is not pending exact adoption"
                )
            trade = pending[1]
            if _snapshot_open_trade(trade) != snapshot:
                raise ValueError(
                    f"order id {snapshot.ib_order_id} broker terms drifted before adoption"
                )
            approved[snapshot.ib_order_id] = _RegisteredOrder(
                trade.contract,
                trade.order,
                _mutation_authority(snapshot),
            )
        self._registry.update(approved)
        for order_id in approved:
            self._pending_adoptions.pop(order_id, None)

    def revoke_adoptions(self) -> None:
        """Remove all mutation authority after any failed reconciliation pass."""
        self._registry.clear()
        self._pending_adoptions.clear()

    async def open_order_refs(self) -> list[tuple[int, str | None, str]]:
        """(ib_order_id, orderRef, status) for every open order at IBKR.
        Minimal surface for now; IBK-128 reconciliation expands on it."""
        await self._client.ensure_connected()
        trades = await self._client.ib.reqAllOpenOrdersAsync()
        if not isinstance(trades, (list, tuple)):
            raise ValueError("open-order snapshot is malformed")
        return [_validated_open_trade(trade) for trade in trades]

    # -- event translation -----------------------------------------------------------

    def _report_callback_error(
        self, kind: BrokerCallbackKind, error: Exception
    ) -> None:
        handler = self._callback_error_handler
        if handler is None:
            log.critical(
                "broker %s callback failed without a safety handler",
                kind,
                exc_info=(type(error), error, error.__traceback__),
            )
            return
        try:
            handler(kind, error)
        except Exception:
            log.critical(
                "broker %s callback safety handler also failed",
                kind,
                exc_info=True,
            )

    def _emit_callbacks(
        self,
        kind: BrokerCallbackKind,
        callbacks: Sequence[Callable[[_CallbackPayload], None]],
        payload: _CallbackPayload,
    ) -> None:
        for callback in tuple(callbacks):
            try:
                callback(payload)
            except Exception as exc:  # noqa: BLE001 -- broker boundary never raises
                self._report_callback_error(kind, exc)

    def _to_order_status_update(self, trade: Trade) -> OrderStatusUpdate:
        raw_order_id = trade.order.orderId
        raw_perm_id = trade.order.permId
        raw_ref = trade.order.orderRef
        raw_status = trade.orderStatus.status
        raw_filled = trade.orderStatus.filled
        raw_remaining = trade.orderStatus.remaining
        raw_avg = trade.orderStatus.avgFillPrice
        if type(raw_order_id) is not int or raw_order_id < 0:
            raise ValueError("order-status identity is malformed")
        if raw_perm_id is not None and (
            type(raw_perm_id) is not int or raw_perm_id < 0
        ):
            raise ValueError("order-status permanent identity is malformed")
        if raw_ref is not None and not isinstance(raw_ref, str):
            raise ValueError("order-status reference is malformed")
        ref = raw_ref or None
        if not isinstance(raw_status, str) or not raw_status.strip():
            raise ValueError("order-status value is malformed")
        numeric = (raw_filled, raw_remaining)
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0
            or not float(value).is_integer()
            for value in numeric
        ):
            raise ValueError("order-status quantities are malformed")
        if raw_avg is not None and (
            not isinstance(raw_avg, (int, float))
            or isinstance(raw_avg, bool)
            or not math.isfinite(float(raw_avg))
            # IBKR reports a combo's net credit as a negative BAG price.
            # Prices for individual option legs must remain nonnegative.
            or (raw_avg < 0 and trade.contract.secType != "BAG")
        ):
            raise ValueError("order-status average fill price is malformed")
        avg_fill_price: float | None = None
        if raw_avg not in (None, 0):
            avg_fill_price = float(raw_avg)
        return OrderStatusUpdate(
            ib_order_id=raw_order_id,
            perm_id=None if raw_perm_id in (None, 0) else raw_perm_id,
            order_ref=ref,
            status=raw_status,
            filled=float(raw_filled),
            remaining=float(raw_remaining),
            avg_fill_price=avg_fill_price,
        )

    def _handle_order_status(self, trade: Trade) -> None:
        try:
            update = self._to_order_status_update(trade)
        except Exception as exc:  # noqa: BLE001 -- broker boundary never raises
            self._report_callback_error("orderStatus", exc)
            return
        self._emit_callbacks("orderStatus", self._status_callbacks, update)

    def _to_execution_fill(
        self, fill: Fill, *, fallback_ref: str | None = None
    ) -> ExecutionFill:
        execution = fill.execution
        if not isinstance(execution.time, datetime):
            raise ValueError("execution timestamp is missing or malformed")
        ts = execution.time
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        raw_shares = execution.shares
        if (
            not isinstance(raw_shares, (int, float))
            or isinstance(raw_shares, bool)
        ):
            raise ValueError("execution quantity is malformed")
        shares = float(raw_shares)
        if not math.isfinite(shares) or shares <= 0 or not shares.is_integer():
            raise ValueError("execution quantity must be a positive whole contract count")
        raw_con_id = fill.contract.conId
        sec_type = fill.contract.secType
        if sec_type not in {"OPT", "BAG"}:
            raise ValueError(f"unknown execution security type {sec_type!r}")
        if (
            type(raw_con_id) is not int
            or raw_con_id < (0 if sec_type == "BAG" else 1)
        ):
            raise ValueError("execution contract identity is malformed")
        side = _IB_SIDE.get(execution.side, execution.side)
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"execution side is malformed: {execution.side!r}")
        raw_price = execution.price
        if (
            not isinstance(raw_price, (int, float))
            or isinstance(raw_price, bool)
        ):
            raise ValueError("execution price is malformed")
        price = float(raw_price)
        if not math.isfinite(price) or (sec_type == "OPT" and price < 0):
            raise ValueError(
                "execution price must be finite and nonnegative for option legs"
            )
        if not isinstance(execution.execId, str) or not execution.execId.strip():
            raise ValueError("execution ID is missing")
        raw_order_id = execution.orderId
        if type(raw_order_id) is not int or raw_order_id < 0:
            raise ValueError("execution order identity is malformed")
        raw_order_ref = execution.orderRef
        if raw_order_ref is not None and not isinstance(raw_order_ref, str):
            raise ValueError("execution order reference is malformed")
        if fallback_ref is not None and not isinstance(fallback_ref, str):
            raise ValueError("fallback order reference is malformed")
        order_ref = raw_order_ref or fallback_ref or None
        report = fill.commissionReport
        commission: float | None = None
        if report is not None and not _empty_commission_placeholder(report):
            if (
                not isinstance(report.execId, str)
                or not report.execId.strip()
                or report.execId != execution.execId
            ):
                raise ValueError("execution commission identity is malformed")
            if (
                not isinstance(report.commission, (int, float))
                or isinstance(report.commission, bool)
                or not math.isfinite(float(report.commission))
            ):
                raise ValueError("execution commission is malformed")
            commission = float(report.commission)
        return ExecutionFill(
            ib_order_id=raw_order_id,
            order_ref=order_ref,
            exec_id=execution.execId,
            side=side,
            price=price,
            qty=int(shares),
            ts=ts,
            con_id=raw_con_id,
            sec_type=sec_type,
            commission=commission,
        )

    @staticmethod
    def _is_manual_foreign_execution(
        fill: Fill,
        *,
        fallback_ref: str | None = None,
    ) -> bool:
        """Whether a known non-options execution is clearly not bot-owned.

        The paper account can contain manual futures/stock activity. Those
        executions must not poison OptionsBot's option-ledger reconciliation,
        while a non-option fill carrying an ``obot-*`` reference still reaches
        strict validation and trips the safety boundary.
        """
        try:
            sec_type = fill.contract.secType
            raw_ref = fill.execution.orderRef
        except AttributeError:
            return False
        if sec_type not in _KNOWN_FOREIGN_EXECUTION_TYPES:
            return False
        if raw_ref is not None and not isinstance(raw_ref, str):
            return False
        if fallback_ref is not None and not isinstance(fallback_ref, str):
            return False
        return ledger_row_id_from_ref(raw_ref or fallback_ref) is None

    async def adopt_open_orders(self) -> list[OpenOrderSnapshot]:
        """Snapshot option-relevant open orders and stage canonical bot rows.

        All prior mutation authority is revoked before the first await. Exact
        reconciliation must later authorize a snapshot batch atomically.
        Clearly manual, known non-option orders are outside OptionsBot's book
        and are discarded before broker-identity validation. TWS can report
        several such external/OCA orders with the placeholder ``orderId=0``;
        treating that representation as a duplicate bot identity would
        incorrectly halt options execution. A non-option order carrying a
        canonical ``obot-*`` reference remains in the strict path and fails
        closed.
        """
        self.revoke_adoptions()
        await self._client.ensure_connected()
        self._ensure_subscribed()
        trades = await self._client.ib.reqAllOpenOrdersAsync()
        if not isinstance(trades, (list, tuple)):
            raise ValueError("open-order snapshot is malformed")
        adopted: list[OpenOrderSnapshot] = []
        pending: dict[int, tuple[OpenOrderSnapshot, Any]] = {}
        seen_order_ids: set[int] = set()
        seen_row_ids: set[int] = set()
        for trade in trades:
            snapshot = _snapshot_open_trade(trade)
            order_id = snapshot.ib_order_id
            ref = snapshot.order_ref
            row_id = ledger_row_id_from_ref(ref)
            sec_type = getattr(getattr(trade, "contract", None), "secType", None)
            if row_id is None and sec_type in _KNOWN_FOREIGN_EXECUTION_TYPES:
                continue
            if order_id in seen_order_ids or (row_id is not None and row_id in seen_row_ids):
                raise ValueError("open-order snapshot contains duplicate broker identity")
            seen_order_ids.add(order_id)
            if row_id is not None:
                seen_row_ids.add(row_id)
                pending[order_id] = (snapshot, trade)
            adopted.append(snapshot)
        self._pending_adoptions = pending
        return adopted

    async def recent_executions(self) -> list[ExecutionFill]:
        """Today's executions for this account (reqExecutions), translated.

        Used by reconciliation to replay fills missed while the daemon was
        down; record_fill's execId dedupe makes the replay idempotent.
        """
        await self._client.ensure_connected()
        from datetime import timedelta

        from ib_async import ExecutionFilter

        # 3-day lookback (IBKR serves ~7 days max): an empty filter returns
        # TODAY only, which would hide weekend-outage fills from
        # reconciliation's mismatch detection (Opus I1). Timezone slop of a
        # few hours is irrelevant at this window size.
        since = (datetime.now(UTC) - timedelta(days=3)).strftime("%Y%m%d %H:%M:%S")
        fills_ = await self._client.ib.reqExecutionsAsync(ExecutionFilter(time=since))
        return [
            self._to_execution_fill(fill)
            for fill in fills_
            if not self._is_manual_foreign_execution(fill)
        ]

    def _handle_exec_details(self, trade: Trade, fill: Fill) -> None:
        if self._is_manual_foreign_execution(
            fill,
            fallback_ref=trade.order.orderRef,
        ):
            return
        try:
            record = self._to_execution_fill(
                fill,
                fallback_ref=trade.order.orderRef,
            )
        except Exception as exc:  # noqa: BLE001 -- broker boundary never raises
            self._report_callback_error("execDetails", exc)
            return
        self._emit_callbacks("execDetails", self._fill_callbacks, record)

    def _handle_commission(
        self, trade: Trade, fill: Fill | None, report: CommissionReport
    ) -> None:
        try:
            if not isinstance(report.execId, str) or not report.execId.strip():
                raise ValueError("commission execution ID is malformed")
            if (
                not isinstance(report.commission, (int, float))
                or isinstance(report.commission, bool)
                or not math.isfinite(float(report.commission))
            ):
                raise ValueError("commission amount is malformed")
            update = CommissionUpdate(
                exec_id=report.execId, commission=float(report.commission)
            )
        except Exception as exc:  # noqa: BLE001 -- broker boundary never raises
            self._report_callback_error("commissionReport", exc)
            return
        self._emit_callbacks("commissionReport", self._commission_callbacks, update)
