"""Strategy registry.

Tasks 1-4 ship 12 of the planned 16 strategies. Later tasks (IBK-41..44) extend
``_STRATEGIES`` to the full set.
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
