"""Strategy registry.

Tasks 1-5 ship 14 of the planned 16 strategies. Later tasks (IBK-43..44)
extend ``_STRATEGIES`` to the full set.
"""

from optionsbot.strategies.base import (
    Leg,
    Strategy,
    StrategySnapshot,
    StrategySuggestion,
)
from optionsbot.strategies.calendar import CalendarSpread, DiagonalSpread
from optionsbot.strategies.iron_butterfly import IronButterfly
from optionsbot.strategies.iron_condor import IronCondor
from optionsbot.strategies.stock_legs import CashSecuredPut, CoveredCall
from optionsbot.strategies.straddles import (
    LongStraddle,
    LongStrangle,
    ShortStraddle,
    ShortStrangle,
)
from optionsbot.strategies.verticals import (
    BearCallSpread,
    BearPutSpread,
    BullCallSpread,
    BullPutSpread,
)

_STRATEGIES: tuple[Strategy, ...] = (
    IronCondor(),
    IronButterfly(),
    BullPutSpread(),
    BearCallSpread(),
    BullCallSpread(),
    BearPutSpread(),
    LongStraddle(),
    LongStrangle(),
    ShortStraddle(),
    ShortStrangle(),
    CalendarSpread(),
    DiagonalSpread(),
    CoveredCall(),
    CashSecuredPut(),
)

_BY_NAME: dict[str, Strategy] = {s.name: s for s in _STRATEGIES}


def all_strategies() -> tuple[Strategy, ...]:
    return _STRATEGIES


def get_strategy(name: str) -> Strategy:
    return _BY_NAME[name]


__all__ = [
    "BearCallSpread",
    "BearPutSpread",
    "BullCallSpread",
    "BullPutSpread",
    "CalendarSpread",
    "CashSecuredPut",
    "CoveredCall",
    "DiagonalSpread",
    "IronButterfly",
    "IronCondor",
    "Leg",
    "LongStraddle",
    "LongStrangle",
    "ShortStraddle",
    "ShortStrangle",
    "Strategy",
    "StrategySnapshot",
    "StrategySuggestion",
    "all_strategies",
    "get_strategy",
]
