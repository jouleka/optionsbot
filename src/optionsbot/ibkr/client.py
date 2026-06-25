"""IBKR connection manager.

Wraps ``ib_async.IB`` so the rest of the codebase has a single point of
contact for connect/disconnect/ensure-connected semantics. Distinct
``client_id`` per process role (MCP=1, daemon=2, exec=3) from settings.
"""

from __future__ import annotations

import asyncio
import logging
from types import TracebackType
from typing import TYPE_CHECKING, Literal

from optionsbot.config import Settings, get_settings

if TYPE_CHECKING:
    from ib_async import IB

log = logging.getLogger(__name__)


ProcessRole = Literal["mcp", "daemon", "cli", "exec"]


class IBKRClient:
    """Single-connection IBKR client with reconnect-with-backoff.

    Usage::

        client = IBKRClient(role="daemon")
        await client.connect()
        ...
        await client.disconnect()

    ``ensure_connected()`` is idempotent and is what callers should use
    before issuing any data request.
    """

    _DEFAULT_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 2.0, 5.0, 15.0, 30.0)

    def __init__(
        self,
        role: ProcessRole = "cli",
        settings: Settings | None = None,
        ib: IB | None = None,  # injected for tests
        backoff_seconds: tuple[float, ...] | None = None,
    ) -> None:
        self._role = role
        self._settings = settings if settings is not None else get_settings()
        if ib is None:
            from ib_async import IB as _IB
            ib = _IB()
        self._ib = ib
        self._backoff = (
            backoff_seconds if backoff_seconds is not None else self._DEFAULT_BACKOFF_SECONDS
        )
        self._connect_lock = asyncio.Lock()

    @property
    def ib(self) -> IB:
        """Underlying ib_async.IB instance. Use sparingly; prefer this client's helpers."""
        return self._ib

    @property
    def role(self) -> ProcessRole:
        return self._role

    @property
    def settings(self) -> Settings:
        """Read-only access to the Settings instance this client was bound to."""
        return self._settings

    @property
    def is_connected(self) -> bool:
        """True if the underlying ib_async client has a live connection."""
        return bool(self._ib.isConnected())

    def _client_id(self) -> int:
        if self._role == "daemon":
            return self._settings.ibkr.client_id_daemon
        if self._role == "exec":
            # IBK-125: order events are only delivered to the placing clientId.
            return self._settings.ibkr.client_id_exec
        # mcp and cli share an id-space; the cli path is short-lived and unlikely to collide
        return self._settings.ibkr.client_id_mcp

    async def connect(self) -> None:
        """Connect to IB Gateway with exponential backoff on failure."""
        async with self._connect_lock:
            if self._ib.isConnected():
                return
            host = self._settings.ibkr.host
            port = self._settings.ibkr.port
            client_id = self._client_id()
            log.info(
                "Connecting to IB Gateway "
                "host=%s port=%s client_id=%s paper=%s role=%s",
                host, port, client_id, self._settings.ibkr.paper, self._role,
            )
            last_exc: Exception | None = None
            for delay in (0.0, *self._backoff):
                if delay:
                    log.warning("Reconnect backoff %.1fs", delay)
                    await asyncio.sleep(delay)
                try:
                    await self._ib.connectAsync(host=host, port=port, clientId=client_id)
                    # IBK-122: in paper mode request the configured market-data
                    # type (default 3 = delayed-streaming; 1 = live once real-time
                    # data is shared into the paper account).
                    if self._settings.ibkr.paper:
                        self._ib.reqMarketDataType(self._settings.ibkr.market_data_type)
                    log.info("Connected to IB Gateway")
                    return
                except Exception as e:  # noqa: BLE001 -- connection failures are heterogeneous
                    last_exc = e
                    log.warning("IB Gateway connect failed: %s", e)
            attempts = len(self._backoff) + 1
            raise ConnectionError(
                f"Could not connect to IB Gateway at {host}:{port} after {attempts} attempts"
            ) from last_exc

    async def ensure_connected(self) -> None:
        if not self._ib.isConnected():
            await self.connect()

    async def disconnect(self) -> None:
        if self._ib.isConnected():
            log.info("Disconnecting from IB Gateway")
            self._ib.disconnect()

    async def __aenter__(self) -> IBKRClient:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.disconnect()
