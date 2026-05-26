"""Strategy registry.

Task 1 of IBK-4 ships only the Strategy ABC + Iron Condor as the canonical
example. Later tasks (IBK-30..44) expand the registry to all 16 strategies.
"""

from optionsbot.strategies.base import (
    Leg,
    Strategy,
    StrategySnapshot,
    StrategySuggestion,
)
from optionsbot.strategies.iron_condor import IronCondor

__all__ = [
    "IronCondor",
    "Leg",
    "Strategy",
    "StrategySnapshot",
    "StrategySuggestion",
]
