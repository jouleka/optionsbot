"""Tests for OrderClient (IBK-125) against a stubbed ib_async.IB."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from optionsbot.config import Settings
from optionsbot.ibkr.client import IBKRClient
from optionsbot.ibkr.contracts import ContractResolver
from optionsbot.ibkr.orders import OrderClient, _validated_open_trade
from optionsbot.ibkr.types import (
    CommissionUpdate,
    ExecutionFill,
    MarginPreview,
    OrderStatusUpdate,
    PlacedOrder,
)

CONDOR_LEGS: list[dict[str, Any]] = [
    {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
     "strike": 580.0, "right": "P", "quantity": 1},
    {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
     "strike": 575.0, "right": "P", "quantity": 1},
    {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
     "strike": 620.0, "right": "C", "quantity": 1},
    {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
     "strike": 625.0, "right": "C", "quantity": 1},
]

CONID_BY_SPEC = {
    ("20260717", 580.0, "P"): 1580,
    ("20260717", 575.0, "P"): 1575,
    ("20260717", 620.0, "C"): 1620,
    ("20260717", 625.0, "C"): 1625,
}


@pytest.fixture()
def order_settings() -> Settings:
    s = Settings()  # paper=True, port=4002 — interlock passes by default
    return s


@pytest.fixture()
def order_ib(mock_ib: MagicMock) -> MagicMock:
    """mock_ib specialized for order flows."""
    mock_ib.isConnected = MagicMock(return_value=True)

    async def fake_qualify(*contracts: Any) -> list[Any]:
        out = []
        for c in contracts:
            key = (c.lastTradeDateOrContractMonth, c.strike, c.right)
            c.conId = CONID_BY_SPEC.get(key, 0)
            c.multiplier = "100"
            c.currency = "USD"
            out.append(c if c.conId else None)
        return out

    mock_ib.qualifyContractsAsync = AsyncMock(side_effect=fake_qualify)

    next_id = iter(range(7, 100))

    def fake_place(contract: Any, order: Any) -> Any:
        from ib_async import Trade

        if not order.orderId:
            order.orderId = next(next_id)
        return Trade(contract=contract, order=order)

    mock_ib.placeOrder = MagicMock(side_effect=fake_place)
    mock_ib.cancelOrder = MagicMock(return_value=None)
    mock_ib.whatIfOrderAsync = AsyncMock()
    # Event hooks: real ib_async IB exposes Event objects supporting +=.
    mock_ib.orderStatusEvent = MagicMock()
    mock_ib.execDetailsEvent = MagicMock()
    mock_ib.commissionReportEvent = MagicMock()
    return mock_ib


@pytest.fixture()
def order_client(order_settings: Settings, order_ib: MagicMock) -> OrderClient:
    client = IBKRClient(role="exec", settings=order_settings, ib=order_ib)
    resolver = ContractResolver(client)
    order_client = OrderClient(client, resolver)
    order_client.on_callback_error(lambda _kind, _error: None)
    return order_client


# --- role / config -----------------------------------------------------------


def test_exec_role_uses_client_id_3(order_settings: Settings) -> None:
    client = IBKRClient(role="exec", settings=order_settings)
    assert client._client_id() == 3  # noqa: SLF001 — pinning the role mapping


# --- combo building + placement ----------------------------------------------


async def test_place_condor_builds_guaranteed_smart_bag(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    placed = await order_client.place_combo_limit(
        "SPY", CONDOR_LEGS, quantity=2, limit_price=-1.55, order_ref="obot-7",
    )
    assert isinstance(placed, PlacedOrder)
    contract, order = order_ib.placeOrder.call_args.args
    assert contract.secType == "BAG"
    assert contract.symbol == "SPY"
    assert contract.currency == "USD"
    assert contract.exchange == "SMART"
    legs = contract.comboLegs
    assert [(leg.conId, leg.action, leg.ratio, leg.exchange) for leg in legs] == [
        (1580, "SELL", 1, "SMART"),
        (1575, "BUY", 1, "SMART"),
        (1620, "SELL", 1, "SMART"),
        (1625, "BUY", 1, "SMART"),
    ]
    assert order.action == "BUY"  # buy the bag; credit = negative net price
    assert order.totalQuantity == 2
    assert order.lmtPrice == -1.55
    assert order.orderRef == "obot-7"
    assert order.tif == "DAY"
    assert placed.ib_order_id == order.orderId
    assert placed.leg_contracts == (
        (1580, 100, "USD"),
        (1575, 100, "USD"),
        (1620, 100, "USD"),
        (1625, 100, "USD"),
    )


async def test_stk_legs_are_filtered_out(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    covered_call = [
        {"symbol": "SPY", "side": "buy", "sec_type": "STK", "quantity": 100,
         "expiry": None, "strike": None, "right": None},
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
         "strike": 620.0, "right": "C", "quantity": 1},
    ]
    await order_client.place_combo_limit(
        "SPY", covered_call, quantity=1, limit_price=-1.20, order_ref="obot-8",
    )
    contract, order = order_ib.placeOrder.call_args.args
    # Single remaining OPT leg -> plain qualified Option, not a 1-leg BAG.
    assert contract.secType == "OPT"
    assert contract.conId == 1620
    assert order.action == "SELL"  # the leg's own side
    assert order.lmtPrice == 1.20  # positive premium for a single leg
    assert order.totalQuantity == 1


async def test_single_long_leg_buys_at_positive_price(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    long_put = [
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
         "strike": 580.0, "right": "P", "quantity": 1},
    ]
    await order_client.place_combo_limit(
        "SPY", long_put, quantity=1, limit_price=2.30, order_ref="obot-9",
    )
    contract, order = order_ib.placeOrder.call_args.args
    assert contract.secType == "OPT"
    assert order.action == "BUY"
    assert order.lmtPrice == 2.30


async def test_single_leg_ratio_multiplies_broker_quantity(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    ratio_leg = [{
        "symbol": "SPY", "side": "sell", "sec_type": "OPT",
        "expiry": "20260717", "strike": 580.0, "right": "P", "quantity": 2,
    }]
    await order_client.place_combo_limit(
        "SPY", ratio_leg, quantity=3, limit_price=-1.0, order_ref="obot-9",
    )
    _, order = order_ib.placeOrder.call_args.args
    assert order.totalQuantity == 6


@pytest.mark.parametrize("ref", ["obot-01", "obot-٠١", "obot-0", "obot-manual"])
async def test_placement_rejects_noncanonical_order_ref(
    order_client: OrderClient, order_ib: MagicMock, ref: str
) -> None:
    with pytest.raises(ValueError, match="canonical"):
        await order_client.place_combo_limit(
            "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.0, order_ref=ref,
        )
    order_ib.placeOrder.assert_not_called()


async def test_unqualified_leg_raises(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    bad_legs = [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
         "strike": 999.0, "right": "P", "quantity": 1},  # not in CONID_BY_SPEC
        {"symbol": "SPY", "side": "buy", "sec_type": "OPT", "expiry": "20260717",
         "strike": 575.0, "right": "P", "quantity": 1},
    ]
    with pytest.raises(ValueError, match="qualify"):
        await order_client.place_combo_limit(
            "SPY", bad_legs, quantity=1, limit_price=-0.5, order_ref="obot-10",
        )
    order_ib.placeOrder.assert_not_called()


async def test_no_option_legs_raises(order_client: OrderClient) -> None:
    with pytest.raises(ValueError, match="option leg"):
        await order_client.place_combo_limit(
            "SPY",
            [{"symbol": "SPY", "side": "buy", "sec_type": "STK", "quantity": 100,
              "expiry": None, "strike": None, "right": None}],
            quantity=1, limit_price=1.0, order_ref="obot-11",
        )


# --- paper interlock ----------------------------------------------------------


async def test_mutations_blocked_off_paper(order_ib: MagicMock) -> None:
    live = Settings()
    live.ibkr.paper = False
    client = IBKRClient(role="exec", settings=live, ib=order_ib)
    oc = OrderClient(client, ContractResolver(client))
    with pytest.raises(RuntimeError, match="paper"):
        await oc.place_combo_limit(
            "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.0, order_ref="obot-12",
        )
    with pytest.raises(RuntimeError, match="paper"):
        await oc.modify_price(7, new_limit_price=-0.9)
    with pytest.raises(RuntimeError, match="paper"):
        await oc.cancel(7)
    order_ib.placeOrder.assert_not_called()


async def test_mutations_blocked_on_live_port(order_ib: MagicMock) -> None:
    live_port = Settings()
    live_port.ibkr.port = 4001  # paper flag True but live Gateway port
    client = IBKRClient(role="exec", settings=live_port, ib=order_ib)
    oc = OrderClient(client, ContractResolver(client))
    with pytest.raises(RuntimeError, match="4001"):
        await oc.place_combo_limit(
            "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.0, order_ref="obot-13",
        )


# --- modify / cancel ------------------------------------------------------------


async def test_modify_price_reuses_same_order_id(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    placed = await order_client.place_combo_limit(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55, order_ref="obot-14",
    )
    await order_client.modify_price(placed.ib_order_id, new_limit_price=-1.45)
    assert order_ib.placeOrder.call_count == 2
    _, first_order = order_ib.placeOrder.call_args_list[0].args
    _, second_order = order_ib.placeOrder.call_args_list[1].args
    assert second_order is first_order  # SAME order object, price-only change
    assert second_order.orderId == placed.ib_order_id
    assert second_order.lmtPrice == -1.45


async def test_modify_single_leg_normalizes_to_positive_premium(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    # Live Error 201: the walk passes signed net targets (credit = negative);
    # a single-leg option must convert to a positive premium on modify just
    # like it does at placement.
    short_put = [
        {"symbol": "SPY", "side": "sell", "sec_type": "OPT", "expiry": "20260717",
         "strike": 580.0, "right": "P", "quantity": 1},
    ]
    placed = await order_client.place_combo_limit(
        "SPY", short_put, quantity=1, limit_price=-6.15, order_ref="obot-16",
    )
    await order_client.modify_price(placed.ib_order_id, new_limit_price=-6.14)
    _, modified_order = order_ib.placeOrder.call_args_list[1].args
    assert modified_order.lmtPrice == pytest.approx(+6.14)  # positive premium


async def test_modify_bag_keeps_signed_limit(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    placed = await order_client.place_combo_limit(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55, order_ref="obot-17",
    )
    await order_client.modify_price(placed.ib_order_id, new_limit_price=-1.45)
    _, modified_order = order_ib.placeOrder.call_args_list[1].args
    assert modified_order.lmtPrice == pytest.approx(-1.45)  # signed BAG net


async def test_modify_unknown_order_raises(order_client: OrderClient) -> None:
    with pytest.raises(ValueError, match="unknown"):
        await order_client.modify_price(31337, new_limit_price=-1.0)


async def test_cancel_uses_registered_order(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    placed = await order_client.place_combo_limit(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55, order_ref="obot-15",
    )
    await order_client.cancel(placed.ib_order_id)
    (cancelled_order,) = order_ib.cancelOrder.call_args.args
    assert cancelled_order.orderId == placed.ib_order_id


# --- whatIf ---------------------------------------------------------------------


async def test_whatif_single_leg_ratio_uses_total_contract_quantity(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    from ib_async import OrderState

    order_ib.whatIfOrderAsync.return_value = OrderState(initMarginChange="100")
    ratio_leg = [{
        "symbol": "SPY", "side": "sell", "sec_type": "OPT",
        "expiry": "20260717", "strike": 580.0, "right": "P", "quantity": 2,
    }]
    await order_client.whatif_combo(
        "SPY", ratio_leg, quantity=3, limit_price=-1.0,
    )
    _, order = order_ib.whatIfOrderAsync.call_args.args
    assert order.totalQuantity == 6


async def test_whatif_parses_order_state(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    from ib_async import OrderState

    order_ib.whatIfOrderAsync.return_value = OrderState(
        initMarginChange="2345.67",
        maintMarginChange="2100.00",
        equityWithLoanChange="-12.5",
        commission=1.31,
        maxCommission=1.7976931348623157e308,  # IBKR's UNSET_DOUBLE sentinel
        warningText="",
    )
    preview = await order_client.whatif_combo(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55,
    )
    assert isinstance(preview, MarginPreview)
    assert preview.init_margin_change == pytest.approx(2345.67)
    assert preview.maint_margin_change == pytest.approx(2100.00)
    assert preview.equity_with_loan_change == pytest.approx(-12.5)
    assert preview.commission == pytest.approx(1.31)
    assert preview.max_commission is None  # UNSET sentinel -> None
    assert preview.warning is None  # empty string -> None


async def test_whatif_handles_unparseable_margins(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    from ib_async import OrderState

    order_ib.whatIfOrderAsync.return_value = OrderState(warningText="check this")
    preview = await order_client.whatif_combo(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55,
    )
    assert preview.init_margin_change is None
    assert preview.warning == "check this"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
async def test_whatif_rejects_non_finite_margins(
    order_client: OrderClient,
    order_ib: MagicMock,
    value: float,
) -> None:
    from ib_async import OrderState

    order_ib.whatIfOrderAsync.return_value = OrderState(initMarginChange=value)
    preview = await order_client.whatif_combo(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55,
    )
    assert preview.init_margin_change is None


async def test_whatif_tolerates_list_result(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    # ib_async 2.1.0 returns a LIST of OrderState from whatIfOrderAsync
    # despite its annotation — observed against a live paper Gateway.
    from ib_async import OrderState

    order_ib.whatIfOrderAsync.return_value = [OrderState(initMarginChange="100.5")]
    preview = await order_client.whatif_combo(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55,
    )
    assert preview.init_margin_change == pytest.approx(100.5)

    order_ib.whatIfOrderAsync.return_value = []
    empty = await order_client.whatif_combo(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.55,
    )
    assert empty.init_margin_change is None
    assert empty.warning is not None


# --- event translation ------------------------------------------------------------


def _make_trade(order_id: int = 7, ref: str = "obot-7") -> Any:
    from ib_async import Contract, Order, OrderStatus, Trade

    order = Order(orderId=order_id, permId=4242, orderRef=ref)
    status = OrderStatus(
        orderId=order_id, status="Submitted", filled=1.0, remaining=1.0,
        avgFillPrice=1.5,
    )
    return Trade(contract=Contract(secType="BAG", symbol="SPY"), order=order,
                 orderStatus=status)


def test_status_event_translates(order_client: OrderClient) -> None:
    seen: list[OrderStatusUpdate] = []
    order_client.on_status(seen.append)
    order_client._handle_order_status(_make_trade())  # noqa: SLF001
    [update] = seen
    assert update.ib_order_id == 7
    assert update.perm_id == 4242
    assert update.order_ref == "obot-7"
    assert update.status == "Submitted"
    assert update.filled == 1.0
    assert update.remaining == 1.0
    assert update.avg_fill_price == 1.5


def test_malformed_status_event_reports_callback_error_without_raising(
    order_client: OrderClient,
) -> None:
    errors: list[tuple[str, Exception]] = []
    order_client._callback_error_handler = (  # type: ignore[attr-defined]  # noqa: SLF001
        lambda kind, error: errors.append((kind, error))
    )
    trade = _make_trade()
    trade.orderStatus.filled = "one"

    order_client._handle_order_status(trade)  # noqa: SLF001

    [(kind, error)] = errors
    assert kind == "orderStatus"
    assert isinstance(error, ValueError)


def test_failing_status_subscriber_does_not_block_later_subscriber(
    order_client: OrderClient,
) -> None:
    errors: list[tuple[str, Exception]] = []
    seen: list[OrderStatusUpdate] = []
    order_client._callback_error_handler = (  # type: ignore[attr-defined]  # noqa: SLF001
        lambda kind, error: errors.append((kind, error))
    )

    def fail(_update: OrderStatusUpdate) -> None:
        raise RuntimeError("subscriber failed")

    order_client.on_status(fail)
    order_client.on_status(seen.append)
    order_client._handle_order_status(_make_trade())  # noqa: SLF001

    assert len(seen) == 1
    [(kind, error)] = errors
    assert kind == "orderStatus"
    assert isinstance(error, RuntimeError)


def test_malformed_execution_event_reports_callback_error_without_raising(
    order_client: OrderClient,
) -> None:
    from datetime import UTC, datetime

    from ib_async import Contract, Execution, Fill

    errors: list[tuple[str, Exception]] = []
    order_client._callback_error_handler = (  # type: ignore[attr-defined]  # noqa: SLF001
        lambda kind, error: errors.append((kind, error))
    )
    execution = Execution(
        execId="exec-bad", time=datetime(2026, 6, 10, 15, 30, tzinfo=UTC),
        side="BOT", shares="one", price=0.40, orderId=7, orderRef="obot-7",
    )
    fill = Fill(
        contract=Contract(secType="OPT", symbol="SPY", conId=1575),
        execution=execution,
        commissionReport=None,
        time=execution.time,  # type: ignore[arg-type]
    )

    order_client._handle_exec_details(_make_trade(), fill)  # noqa: SLF001

    [(kind, error)] = errors
    assert kind == "execDetails"
    assert isinstance(error, ValueError)


def test_malformed_commission_event_reports_callback_error_without_raising(
    order_client: OrderClient,
) -> None:
    from ib_async import CommissionReport

    errors: list[tuple[str, Exception]] = []
    order_client._callback_error_handler = (  # type: ignore[attr-defined]  # noqa: SLF001
        lambda kind, error: errors.append((kind, error))
    )
    report = CommissionReport(execId="exec-bad", commission="many")

    order_client._handle_commission(_make_trade(), None, report)  # noqa: SLF001

    [(kind, error)] = errors
    assert kind == "commissionReport"
    assert isinstance(error, ValueError)


def test_fill_event_translates_and_normalizes_side(
    order_client: OrderClient,
) -> None:
    from datetime import UTC, datetime

    from ib_async import Contract, Execution, Fill

    seen: list[ExecutionFill] = []
    order_client.on_fill(seen.append)
    execution = Execution(
        execId="0001.aa.01", time=datetime(2026, 6, 10, 15, 30, tzinfo=UTC),
        side="BOT", shares=2.0, price=0.40, permId=4242, orderId=7,
        orderRef="obot-7",
    )
    leg_contract = Contract(secType="OPT", symbol="SPY", conId=1575)
    fill = Fill(
        contract=leg_contract, execution=execution,
        commissionReport=None, time=execution.time,  # type: ignore[arg-type]
    )
    order_client._handle_exec_details(_make_trade(), fill)  # noqa: SLF001
    [record] = seen
    assert record.exec_id == "0001.aa.01"
    assert record.side == "BUY"  # BOT normalized
    assert record.qty == 2
    assert record.price == 0.40
    assert record.con_id == 1575
    assert record.sec_type == "OPT"
    assert record.order_ref == "obot-7"
    assert record.ts.tzinfo is not None


def test_execution_adapter_accepts_negative_bag_credit_price(
    order_client: OrderClient,
) -> None:
    from datetime import UTC, datetime

    from ib_async import Contract, Execution, Fill

    execution = Execution(
        execId="combo-credit",
        time=datetime(2026, 7, 14, 13, 42, tzinfo=UTC),
        side="BOT",
        shares=1.0,
        price=-1.95,
        orderId=171,
        orderRef="obot-12",
    )
    fill = Fill(
        contract=Contract(secType="BAG", symbol="SPY", conId=0),
        execution=execution,
        commissionReport=None,
        time=execution.time,
    )

    translated = order_client._to_execution_fill(fill)  # noqa: SLF001

    assert translated.sec_type == "BAG"
    assert translated.price == pytest.approx(-1.95)


def test_execution_adapter_rejects_negative_option_leg_price(
    order_client: OrderClient,
) -> None:
    from datetime import UTC, datetime

    from ib_async import Contract, Execution, Fill

    execution = Execution(
        execId="negative-option-leg",
        time=datetime(2026, 7, 14, 13, 42, tzinfo=UTC),
        side="BOT",
        shares=1.0,
        price=-1.95,
        orderId=171,
        orderRef="obot-12",
    )
    fill = Fill(
        contract=Contract(secType="OPT", symbol="SPY", conId=1580),
        execution=execution,
        commissionReport=None,
        time=execution.time,
    )

    with pytest.raises(ValueError, match="nonnegative for option legs"):
        order_client._to_execution_fill(fill)  # noqa: SLF001


def test_execution_adapter_rejects_mismatched_commission_identity(
    order_client: OrderClient,
) -> None:
    from datetime import UTC, datetime

    from ib_async import CommissionReport, Contract, Execution, Fill

    execution = Execution(
        execId="exec-1",
        time=datetime(2026, 6, 10, 15, 30, tzinfo=UTC),
        side="BOT",
        shares=1.0,
        price=0.40,
        orderId=7,
        orderRef="obot-7",
    )
    fill = Fill(
        contract=Contract(secType="OPT", symbol="SPY", conId=1580),
        execution=execution,
        commissionReport=CommissionReport(execId="other-exec", commission=0.66),
        time=execution.time,
    )

    with pytest.raises(ValueError):
        order_client._to_execution_fill(fill)  # noqa: SLF001


def test_execution_adapter_treats_empty_commission_report_as_unbundled(
    order_client: OrderClient,
) -> None:
    from datetime import UTC, datetime

    from ib_async import CommissionReport, Contract, Execution, Fill

    execution = Execution(
        execId="exec-1",
        time=datetime(2026, 6, 10, 15, 30, tzinfo=UTC),
        side="BOT",
        shares=1.0,
        price=0.40,
        orderId=7,
        orderRef="obot-7",
    )
    fill = Fill(
        contract=Contract(secType="OPT", symbol="SPY", conId=1580),
        execution=execution,
        commissionReport=CommissionReport(),
        time=execution.time,
    )

    translated = order_client._to_execution_fill(fill)  # noqa: SLF001

    assert translated.commission is None


@pytest.mark.parametrize(
    ("shares", "con_id"),
    [(1.5, 1580), (1.0, 1580.5)],
)
def test_execution_adapter_rejects_fractional_quantity_or_contract_identity(
    order_client: OrderClient,
    shares: float,
    con_id: float,
) -> None:
    from datetime import UTC, datetime

    from ib_async import Contract, Execution, Fill

    execution = Execution(
        execId="bad-execution",
        time=datetime(2026, 6, 10, 15, 30, tzinfo=UTC),
        side="BOT",
        shares=shares,
        price=0.40,
        orderId=7,
        orderRef="obot-7",
    )
    fill = Fill(
        contract=Contract(secType="OPT", symbol="SPY", conId=con_id),  # type: ignore[arg-type]
        execution=execution,
        commissionReport=None,
        time=execution.time,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError):
        order_client._to_execution_fill(fill)  # noqa: SLF001


@pytest.mark.parametrize(
    ("field", "malformed_value"),
    [("shares", "1"), ("price", "0.40"), ("orderRef", False)],
)
def test_execution_adapter_rejects_wrong_type_provider_values(
    order_client: OrderClient,
    field: str,
    malformed_value: object,
) -> None:
    from datetime import UTC, datetime

    from ib_async import Contract, Execution, Fill

    execution = Execution(
        execId="typed-execution",
        time=datetime(2026, 6, 10, 15, 30, tzinfo=UTC),
        side="BOT",
        shares=1.0,
        price=0.40,
        orderId=7,
        orderRef="obot-7",
    )
    setattr(execution, field, malformed_value)
    fill = Fill(
        contract=Contract(secType="OPT", symbol="SPY", conId=1580),
        execution=execution,
        commissionReport=None,
        time=execution.time,
    )

    with pytest.raises(ValueError):
        order_client._to_execution_fill(fill)  # noqa: SLF001


def test_open_order_adapter_rejects_false_reference() -> None:
    from ib_async import Contract, Order, OrderStatus, Trade

    trade = Trade(
        contract=Contract(secType="OPT", symbol="SPY"),
        order=Order(orderId=7, orderRef=False),  # type: ignore[arg-type]
        orderStatus=OrderStatus(orderId=7, status="Submitted"),
    )
    with pytest.raises(ValueError):
        _validated_open_trade(trade)


@pytest.mark.parametrize(
    ("field", "malformed_value"),
    [
        ("conId", 1580.5),
        ("multiplier", ""),
        ("multiplier", "50"),
        ("currency", ""),
        ("currency", "EUR"),
    ],
)
async def test_placement_rejects_malformed_qualified_contract_before_broker_call(
    order_client: OrderClient,
    order_ib: MagicMock,
    field: str,
    malformed_value: object,
) -> None:
    from ib_async import Option

    leg = CONDOR_LEGS[0]
    spec = (leg["expiry"], leg["strike"], leg["right"])
    malformed = Option("SPY", leg["expiry"], leg["strike"], leg["right"], "SMART")
    malformed.conId = 1580
    malformed.multiplier = "100"
    malformed.currency = "USD"
    setattr(malformed, field, malformed_value)
    order_client._resolver.qualify_options = AsyncMock(  # noqa: SLF001
        return_value={spec: malformed}
    )

    with pytest.raises(ValueError):
        await order_client.place_combo_limit(
            "SPY",
            [leg],
            quantity=1,
            limit_price=1.0,
            order_ref="obot-7",
        )

    order_ib.placeOrder.assert_not_called()


def test_status_event_tolerates_none_perm_id(order_client: OrderClient) -> None:
    # Order.permId defaults to 0 but the dataclass accepts None; a None must
    # not blow up the handler (the status update would be silently dropped).
    from ib_async import Contract, Order, OrderStatus, Trade

    seen: list[OrderStatusUpdate] = []
    order_client.on_status(seen.append)
    trade = Trade(
        contract=Contract(secType="BAG", symbol="SPY"),
        order=Order(orderId=7, permId=None, orderRef="obot-7"),  # type: ignore[arg-type]
        orderStatus=OrderStatus(orderId=7, status="Submitted", filled=0, remaining=1),
    )
    order_client._handle_order_status(trade)  # noqa: SLF001
    [update] = seen
    assert update.perm_id is None


async def test_adopt_open_orders_rebinds_registry_for_our_orders_only(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    from ib_async import ComboLeg, Contract, Order, OrderStatus, Trade

    ours = Trade(
        contract=Contract(
            secType="BAG",
            symbol="SPY",
            currency="USD",
            exchange="SMART",
            comboLegs=[
                ComboLeg(
                    conId=1580,
                    ratio=1,
                    action="SELL",
                    exchange="SMART",
                ),
                ComboLeg(
                    conId=1575,
                    ratio=1,
                    action="BUY",
                    exchange="SMART",
                ),
            ],
        ),
        order=Order(
            orderId=44,
            permId=99,
            orderRef="obot-12",
            action="BUY",
            totalQuantity=1,
            orderType="LMT",
            lmtPrice=-1.0,
            tif="DAY",
        ),
        orderStatus=OrderStatus(orderId=44, status="Submitted"),
    )
    manual = Trade(
        contract=Contract(secType="OPT", symbol="SPY"),
        order=Order(orderId=0, permId=7, orderRef=""),
        orderStatus=OrderStatus(orderId=0, status="Submitted"),
    )
    order_ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[ours, manual])

    adopted = await order_client.adopt_open_orders()
    ours_snapshot = next(row for row in adopted if row.order_ref == "obot-12")
    assert ours_snapshot.ib_order_id == 44
    assert ours_snapshot.status == "Submitted"
    assert ours_snapshot.sec_type == "BAG"
    assert ours_snapshot.symbol == "SPY"
    assert ours_snapshot.currency == "USD"
    assert ours_snapshot.exchange == "SMART"
    assert ours_snapshot.combo_legs == (
        (1580, 1, "SELL", "SMART"),
        (1575, 1, "BUY", "SMART"),
    )
    assert ours_snapshot.order_action == "BUY"
    assert ours_snapshot.total_quantity == 1
    assert ours_snapshot.order_type == "LMT"
    assert ours_snapshot.tif == "DAY"
    assert ours_snapshot.limit_price == pytest.approx(-1.0)
    manual_snapshot = next(row for row in adopted if row.order_ref is None)
    assert manual_snapshot.ib_order_id == 0
    # Snapshot collection alone is not authorization to mutate the order.
    with pytest.raises(ValueError, match="unknown"):
        await order_client.modify_price(44, new_limit_price=-1.10)
    order_client.authorize_adoptions((ours_snapshot,))
    # Exact reconciliation has now authorized this order for mutation.
    await order_client.modify_price(44, new_limit_price=-1.10)
    order_ib.placeOrder.assert_called_once()
    # ...but a manual TWS order (orderId 0) must never be modifiable.
    with pytest.raises(ValueError, match="unknown|identity"):
        await order_client.modify_price(0, new_limit_price=-1.0)


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("contract", "symbol", "QQQ"),
        ("contract", "currency", "EUR"),
        ("contract", "exchange", "CBOE"),
        ("contract", "conId", 9991),
        ("contract", "multiplier", "50"),
        ("contract", "lastTradeDateOrContractMonth", "20260821"),
        ("contract", "strike", 590.0),
        ("contract", "right", "C"),
        ("order", "orderId", 56),
        ("order", "orderRef", "obot-2"),
        ("order", "action", "BUY"),
        ("order", "totalQuantity", 99),
        ("order", "orderType", "MKT"),
        ("order", "lmtPrice", 2.0),
        ("order", "tif", "GTC"),
        ("status", "status", "PreSubmitted"),
    ],
)
async def test_authorization_resnapshots_all_mutable_broker_terms(
    order_client: OrderClient,
    order_ib: MagicMock,
    owner: str,
    field: str,
    value: object,
) -> None:
    from ib_async import Contract, Order, OrderStatus, Trade

    trade = Trade(
        contract=Contract(
            secType="OPT", symbol="SPY", currency="USD", exchange="SMART",
            conId=1580, multiplier="100", lastTradeDateOrContractMonth="20260717",
            strike=580.0, right="P",
        ),
        order=Order(
            orderId=55, orderRef="obot-1", action="SELL", totalQuantity=1,
            orderType="LMT", lmtPrice=1.0, tif="DAY",
        ),
        orderStatus=OrderStatus(orderId=55, status="Submitted"),
    )
    order_ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[trade])
    [snapshot] = await order_client.adopt_open_orders()
    target = {
        "contract": trade.contract,
        "order": trade.order,
        "status": trade.orderStatus,
    }[owner]
    setattr(target, field, value)

    with pytest.raises(ValueError):
        order_client.authorize_adoptions((snapshot,))
    with pytest.raises(ValueError, match="unknown"):
        await order_client.modify_price(55, new_limit_price=2.0)


async def test_authorization_is_atomic_when_later_snapshot_drifts(
    order_client: OrderClient,
    order_ib: MagicMock,
) -> None:
    from ib_async import Contract, Order, OrderStatus, Trade

    def trade(order_id: int, row_id: int, con_id: int) -> Trade:
        return Trade(
            contract=Contract(
                secType="OPT", symbol="SPY", currency="USD", exchange="SMART",
                conId=con_id, multiplier="100",
                lastTradeDateOrContractMonth="20260717", strike=580.0, right="P",
            ),
            order=Order(
                orderId=order_id, orderRef=f"obot-{row_id}", action="SELL",
                totalQuantity=1, orderType="LMT", lmtPrice=1.0, tif="DAY",
            ),
            orderStatus=OrderStatus(orderId=order_id, status="Submitted"),
        )

    first = trade(55, 1, 1580)
    second = trade(56, 2, 1581)
    order_ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[first, second])
    snapshots = tuple(await order_client.adopt_open_orders())
    pending_before = dict(order_client._pending_adoptions)  # noqa: SLF001
    second.order.totalQuantity = 99

    with pytest.raises(ValueError):
        order_client.authorize_adoptions(snapshots)

    assert not order_client._registry  # noqa: SLF001
    assert order_client._pending_adoptions == pending_before  # noqa: SLF001


@pytest.mark.parametrize("mutation", ["modify", "cancel"])
async def test_authorized_order_drift_blocks_later_mutation(
    order_client: OrderClient,
    order_ib: MagicMock,
    mutation: str,
) -> None:
    from ib_async import Contract, Order, OrderStatus, Trade

    trade = Trade(
        contract=Contract(
            secType="OPT", symbol="SPY", currency="USD", exchange="SMART",
            conId=1580, multiplier="100", lastTradeDateOrContractMonth="20260717",
            strike=580.0, right="P",
        ),
        order=Order(
            orderId=55, orderRef="obot-1", action="SELL", totalQuantity=1,
            orderType="LMT", lmtPrice=1.0, tif="DAY",
        ),
        orderStatus=OrderStatus(orderId=55, status="Submitted"),
    )
    order_ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[trade])
    [snapshot] = await order_client.adopt_open_orders()
    order_client.authorize_adoptions((snapshot,))
    trade.order.totalQuantity = 99

    with pytest.raises(ValueError, match="drifted"):
        if mutation == "modify":
            await order_client.modify_price(55, new_limit_price=2.0)
        else:
            await order_client.cancel(55)

    order_ib.placeOrder.assert_not_called()
    order_ib.cancelOrder.assert_not_called()


async def test_later_malformed_open_order_leaves_no_partial_pending_batch(
    order_client: OrderClient,
    order_ib: MagicMock,
) -> None:
    from ib_async import Contract, Order, OrderStatus, Trade

    def trade(order_id: int, row_id: int, *, currency: str = "USD") -> Trade:
        return Trade(
            contract=Contract(
                secType="OPT", symbol="SPY", currency=currency, exchange="SMART",
                conId=1580 + row_id, multiplier="100",
                lastTradeDateOrContractMonth="20260717", strike=580.0, right="P",
            ),
            order=Order(
                orderId=order_id, orderRef=f"obot-{row_id}", action="SELL",
                totalQuantity=1, orderType="LMT", lmtPrice=1.0, tif="DAY",
            ),
            orderStatus=OrderStatus(orderId=order_id, status="Submitted"),
        )

    order_ib.reqAllOpenOrdersAsync = AsyncMock(
        return_value=[trade(55, 1), trade(56, 2, currency="usd")]
    )

    with pytest.raises(ValueError, match="currency"):
        await order_client.adopt_open_orders()

    assert not order_client._pending_adoptions  # noqa: SLF001


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("multiplier", "１００"),
        ("lastTradeDateOrContractMonth", "２０２６０７１７"),
    ],
)
async def test_unicode_numeric_option_terms_are_not_adopted(
    order_client: OrderClient,
    order_ib: MagicMock,
    field: str,
    value: str,
) -> None:
    from ib_async import Contract, Order, OrderStatus, Trade

    contract = Contract(
        secType="OPT", symbol="SPY", currency="USD", exchange="SMART",
        conId=1580, multiplier="100", lastTradeDateOrContractMonth="20260717",
        strike=580.0, right="P",
    )
    setattr(contract, field, value)
    order_ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[Trade(
        contract=contract,
        order=Order(
            orderId=55, orderRef="obot-1", action="SELL", totalQuantity=1,
            orderType="LMT", lmtPrice=1.0, tif="DAY",
        ),
        orderStatus=OrderStatus(orderId=55, status="Submitted"),
    )])

    with pytest.raises(ValueError, match="contract"):
        await order_client.adopt_open_orders()

    assert not order_client._pending_adoptions  # noqa: SLF001


async def test_fresh_snapshot_revokes_stale_registry_authority(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    placed = await order_client.place_combo_limit(
        "SPY", CONDOR_LEGS, quantity=1, limit_price=-1.0, order_ref="obot-1",
    )
    from ib_async import Contract, Order, OrderStatus, Trade

    order_ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[Trade(
        contract=Contract(secType="OPT", symbol="SPY"),
        order=Order(orderId=placed.ib_order_id, orderRef="manual-x"),
        orderStatus=OrderStatus(orderId=placed.ib_order_id, status="Submitted"),
    )])
    await order_client.adopt_open_orders()

    with pytest.raises(ValueError, match="unknown"):
        await order_client.cancel(placed.ib_order_id)


async def test_noncanonical_obot_reference_never_becomes_mutable(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    from ib_async import Contract, Order, OrderStatus, Trade

    trade = Trade(
        contract=Contract(secType="OPT", symbol="SPY"),
        order=Order(orderId=55, orderRef="obot-manual"),
        orderStatus=OrderStatus(orderId=55, status="Submitted"),
    )
    order_ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[trade])

    [snapshot] = await order_client.adopt_open_orders()
    assert snapshot.order_ref == "obot-manual"
    with pytest.raises(ValueError, match="not pending"):
        order_client.authorize_adoptions((snapshot,))
    with pytest.raises(ValueError, match="unknown"):
        await order_client.modify_price(55, new_limit_price=1.0)


@pytest.mark.parametrize("ref", ["obot-01", "obot-٠١", "obot-１２", "obot-0"])
async def test_numeric_alias_reference_never_becomes_pending(
    order_client: OrderClient, order_ib: MagicMock, ref: str
) -> None:
    from ib_async import Contract, Order, OrderStatus, Trade

    trade = Trade(
        contract=Contract(secType="OPT", symbol="SPY"),
        order=Order(orderId=55, orderRef=ref),
        orderStatus=OrderStatus(orderId=55, status="Submitted"),
    )
    order_ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[trade])

    [snapshot] = await order_client.adopt_open_orders()
    assert snapshot.order_ref == ref
    with pytest.raises(ValueError, match="not pending"):
        order_client.authorize_adoptions((snapshot,))


async def test_duplicate_broker_identity_is_not_pending(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    from ib_async import Contract, Order, OrderStatus, Trade

    def trade(con_id: int) -> Any:
        return Trade(
            contract=Contract(
                secType="OPT", symbol="SPY", currency="USD", exchange="SMART",
                conId=con_id, multiplier="100", lastTradeDateOrContractMonth="20260717",
                strike=580.0, right="P",
            ),
            order=Order(
                orderId=55, orderRef="obot-1", action="SELL", totalQuantity=1,
                orderType="LMT", lmtPrice=1.0, tif="DAY",
            ),
            orderStatus=OrderStatus(orderId=55, status="Submitted"),
        )

    order_ib.reqAllOpenOrdersAsync = AsyncMock(return_value=[trade(1580), trade(9991)])

    with pytest.raises(ValueError, match="duplicate"):
        await order_client.adopt_open_orders()
    assert not order_client._pending_adoptions  # noqa: SLF001


async def test_recent_executions_translates_with_commission(
    order_client: OrderClient, order_ib: MagicMock
) -> None:
    from datetime import UTC, datetime

    from ib_async import CommissionReport, Contract, Execution, Fill

    execution = Execution(
        execId="0009.aa.01", time=datetime(2026, 6, 11, 15, 0, tzinfo=UTC),
        side="SLD", shares=1.0, price=1.55, permId=99, orderId=44,
        orderRef="obot-12",
    )
    fill = Fill(
        contract=Contract(secType="OPT", symbol="SPY", conId=1580),
        execution=execution,
        commissionReport=CommissionReport(execId="0009.aa.01", commission=0.66),
        time=execution.time,
    )
    order_ib.reqExecutionsAsync = AsyncMock(return_value=[fill])

    [record] = await order_client.recent_executions()
    # The filter must carry a lookback time — an empty filter returns TODAY
    # only and would hide weekend-outage fills from reconciliation.
    (exec_filter,) = order_ib.reqExecutionsAsync.call_args.args
    assert exec_filter.time  # non-empty lookback
    assert record.exec_id == "0009.aa.01"
    assert record.side == "SELL"
    assert record.order_ref == "obot-12"
    assert record.commission == pytest.approx(0.66)
    assert record.sec_type == "OPT"


def test_commission_event_translates(order_client: OrderClient) -> None:
    from ib_async import CommissionReport

    seen: list[CommissionUpdate] = []
    order_client.on_commission(seen.append)
    report = CommissionReport(execId="0001.aa.01", commission=1.31, currency="USD")
    order_client._handle_commission(_make_trade(), None, report)  # noqa: SLF001
    [update] = seen
    assert update.exec_id == "0001.aa.01"
    assert update.commission == pytest.approx(1.31)
