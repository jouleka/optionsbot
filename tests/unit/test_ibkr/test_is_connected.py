from unittest.mock import MagicMock

from optionsbot.config import Settings
from optionsbot.ibkr.client import IBKRClient


def test_is_connected_delegates_to_ib() -> None:
    c = IBKRClient(role="daemon", settings=Settings())
    c._ib = MagicMock()
    c._ib.isConnected.return_value = True
    assert c.is_connected is True
    c._ib.isConnected.return_value = False
    assert c.is_connected is False
