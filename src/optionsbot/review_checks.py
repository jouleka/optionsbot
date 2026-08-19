"""Canonical names for the seven Hermes entry-review gates.

The Hermes skill uses descriptive public names while the original MCP contract
used shorter internal names.  Normalize at every trust boundary so a complete
review cannot be rejected solely because those two vocabularies differ.
"""

from __future__ import annotations

from collections.abc import Mapping

REQUIRED_ENTRY_CHECKS = frozenset(
    {
        "bot_health",
        "candidate",
        "microstructure",
        "greeks",
        "regime_history",
        "catalysts",
        "account_risk",
    }
)

_ENTRY_CHECK_ALIASES = {
    "bot_health": "bot_health",
    "candidate": "candidate",
    "candidate_definition": "candidate",
    "microstructure": "microstructure",
    "market_microstructure": "microstructure",
    "greeks": "greeks",
    "greeks_and_structure_risk": "greeks",
    "regime_history": "regime_history",
    "regime_and_history": "regime_history",
    "catalysts": "catalysts",
    "account_risk": "account_risk",
    "account_risk_caps": "account_risk",
}


def normalize_entry_checks(value: object) -> dict[str, bool] | None:
    """Return one exact canonical seven-gate map, or ``None`` when malformed."""
    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, bool] = {}
    for raw_name, raw_result in value.items():
        if not isinstance(raw_name, str) or not isinstance(raw_result, bool):
            return None
        name = _ENTRY_CHECK_ALIASES.get(raw_name)
        if name is None or name in normalized:
            return None
        normalized[name] = raw_result
    if set(normalized) != REQUIRED_ENTRY_CHECKS:
        return None
    return normalized


def all_entry_checks_pass(value: object) -> bool:
    """Whether ``value`` is a complete canonical-or-public all-true gate map."""
    normalized = normalize_entry_checks(value)
    return normalized is not None and all(normalized.values())
