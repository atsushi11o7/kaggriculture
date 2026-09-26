"""Land-purchase deadlines keyed by Kaggle's raw zero-based day."""

from __future__ import annotations

# Modal purchase days measured directly from top-player replay observations.
# Retrying after each threshold handles cash-dependent one-day delays.
_SCHEDULE = (("NE", 6, 1000), ("SW", 9, 2000), ("SE", 10, 4000))
_OPERATING_RESERVE = 300


def required_capital(obs: dict, purchases_ahead: int = 0) -> int:
    """Return cash to reserve for a pending scheduled land purchase."""
    player = obs["player"]
    owned = set(obs["farms"][player]["unlocked_quadrants"])
    pending = [entry for entry in _SCHEDULE if entry[0] not in owned]
    if purchases_ahead >= len(pending):
        return 0
    return pending[purchases_ahead][2] + _OPERATING_RESERVE


def next_forced_purchase(obs: dict, available_cash: float | None = None) -> list | None:
    """Return BUY_LAND when the raw observation-day deadline is affordable."""
    player = obs["player"]
    farm = obs["farms"][player]
    owned = set(farm["unlocked_quadrants"])
    day = obs["day"]
    cash = farm["money"] if available_cash is None else available_cash

    for quadrant, eligible_day, price in _SCHEDULE:
        if quadrant in owned:
            continue
        if day >= eligible_day and cash >= price + _OPERATING_RESERVE:
            return ["BUY_LAND"]
        return None
    return None
