"""Day- and shop-conditioned production targets for the opening.

Crop and animal targets are looked up directly from empirically measured
day x demand-level means (DSM/DECEM/Vadim Vasilenko day11-full-land replays,
~1950 player-games from 2026-09-21 through 2026-09-25), rather than a
hand-fit formula. A formula assuming steady growth systematically overshoots
several fields: SHEEP stays flat at 3 all opening without YARN_STORE, and
CARROT/TOMATO stay near zero without their own demand shop.
"""

from __future__ import annotations

from dataclasses import dataclass

from kaggriculture.policy.opening.demand import product_demand
from kaggriculture.rules import constants as C

_HANDS = (6, 6, 6, 8, 9, 9, 10, 11, 11)  # observation days 3..11


@dataclass(frozen=True)
class OpeningTargets:
    """Desired active capacity at the current opening day."""

    hands: int
    crops: dict[str, int]
    animals: dict[str, int]


# Measured mean tile/head counts at 23:00 of each raw observation day,
# keyed by {day: {demand_level: mean}}. demand_level is the field's own
# product demand from unlocked shops (kaggriculture.policy.opening.demand),
# capped at 2. Not every level has replay support at every day (a
# single-product shop like YARN_STORE jumps demand straight from 0 to 2, so
# level 1 is never observed for SHEEP); _lookup falls back to the closest
# level at or below what's requested.
_WHEAT = {
    3: {0: 0.4, 1: 0.4},
    4: {0: 0.0, 1: 0.0},
    5: {0: 0.0, 1: 0.0},
    6: {0: 6.5, 1: 5.8, 2: 4.5},
    7: {0: 9.4, 1: 8.4, 2: 6.5},
    8: {0: 8.7, 1: 8.1, 2: 7.2},
    9: {0: 19.5, 1: 19.8, 2: 20.0},
    10: {0: 25.0, 1: 26.1, 2: 27.6},
    11: {0: 26.7, 1: 28.8, 2: 31.0},
}
_CARROT = {
    3: {0: 0.0, 1: 0.0, 2: 0.0},
    4: {0: 0.0, 1: 0.0, 2: 0.0},
    5: {0: 0.0, 1: 0.0, 2: 0.0},
    6: {0: 0.0, 1: 0.0, 2: 0.2},
    7: {0: 0.0, 1: 0.0, 2: 0.3},
    8: {0: 0.0, 1: 0.0, 2: 0.3},
    9: {0: 0.0, 1: 0.0, 2: 1.5},
    10: {0: 0.0, 1: 0.1, 2: 5.2},
    11: {0: 0.1, 1: 0.1, 2: 8.4},
}
_TOMATO = {
    3: {0: 0.0, 1: 0.0},
    4: {0: 0.0, 1: 0.0},
    5: {0: 0.0, 1: 0.0},
    6: {0: 0.1, 1: 0.1, 2: 0.2},
    7: {0: 0.1, 1: 0.2, 2: 0.2},
    8: {0: 0.1, 1: 0.2, 2: 0.2},
    9: {0: 1.4, 1: 2.3, 2: 3.3},
    10: {0: 1.6, 1: 2.9, 2: 5.0},
    11: {0: 2.5, 1: 4.6, 2: 8.1},
}
_STRAWBERRY = {
    3: {0: 8.8, 1: 8.9},
    4: {0: 9.4, 1: 9.5},
    5: {0: 9.4, 1: 9.5},
    6: {0: 11.1, 1: 17.2, 2: 21.2},
    7: {0: 11.3, 1: 18.1, 2: 23.3},
    8: {0: 11.3, 1: 18.2, 2: 24.3},
    9: {0: 11.5, 1: 16.6, 2: 24.2},
    10: {0: 11.6, 1: 17.6, 2: 27.5},
    11: {0: 12.5, 1: 19.3, 2: 30.2},
}
_MELON = {  # no shop demands melon: demand level is always 0
    3: {0: 10.4},
    4: {0: 10.4},
    5: {0: 10.4},
    6: {0: 10.5},
    7: {0: 10.5},
    8: {0: 10.5},
    9: {0: 10.5},
    10: {0: 4.5},
    11: {0: 0.5},
}
_GOOSE = {
    3: {0: 0.0, 1: 0.0},
    4: {0: 0.0, 1: 0.0},
    5: {0: 0.0, 1: 0.0},
    6: {0: 1.0, 1: 2.3, 2: 4.1},
    7: {0: 1.0, 1: 2.5, 2: 4.8},
    8: {0: 1.1, 1: 2.6, 2: 4.9},
    9: {0: 2.4, 1: 4.2, 2: 6.5},
    10: {0: 3.5, 1: 5.6, 2: 8.1},
    11: {0: 4.4, 1: 6.6, 2: 9.0},
}
_COW = {
    3: {0: 2.0, 1: 2.1},
    4: {0: 2.1, 1: 2.1},
    5: {0: 2.1, 1: 2.1},
    6: {0: 4.5, 1: 6.8, 2: 8.6},
    7: {0: 4.6, 1: 7.5, 2: 9.5},
    8: {0: 4.6, 1: 7.8, 2: 10.9},
    9: {0: 5.6, 1: 8.1, 2: 11.8},
    10: {0: 5.6, 1: 8.2, 2: 12.1},
    11: {0: 5.6, 1: 8.2, 2: 12.1},
}
_SHEEP = {  # WOOL is only demanded by YARN_STORE (single-product, level 2)
    3: {0: 3.0, 2: 3.0},
    4: {0: 3.0, 2: 3.1},
    5: {0: 3.0, 2: 3.1},
    6: {0: 3.0, 2: 7.5},
    7: {0: 3.0, 2: 8.0},
    8: {0: 3.0, 2: 10.5},
    9: {0: 3.0, 2: 9.5},
    10: {0: 3.0, 2: 10.8},
    11: {0: 3.0, 2: 11.0},
}


def _lookup(table: dict[int, dict[int, float]], day: int, level: int) -> int:
    """Return the measured mean for a day, falling back to a nearby level."""
    day = min(max(day, 3), 11)
    levels = table[day]
    if level in levels:
        return round(levels[level])
    lower = [key for key in levels if key <= level]
    return round(levels[max(lower)] if lower else levels[min(levels)])


def for_state(day: int, unlocked_shops: list[str]) -> OpeningTargets:
    """Build targets for a raw observation day and unlocked shop demand."""
    index = min(max(day, 3), 11) - 3
    demand = product_demand(unlocked_shops)

    crops = {
        "WHEAT": _lookup(_WHEAT, day, min(demand["WHEAT"], 2)),
        "CARROT": _lookup(_CARROT, day, min(demand["CARROT"], 2)),
        "TOMATO": _lookup(_TOMATO, day, min(demand["TOMATO"], 2)),
        "STRAWBERRY": _lookup(_STRAWBERRY, day, min(demand["STRAWBERRY"], 2)),
        "MELON": _lookup(_MELON, day, 0),
    }
    animals = {
        "GOOSE": _lookup(_GOOSE, day, min(demand["EGG"], 2)),
        "COW": _lookup(_COW, day, min(demand["MILK"], 2)),
        "SHEEP": _lookup(_SHEEP, day, min(demand["WOOL"], 2)),
    }

    # Never request more occupied tiles than the fully unlocked board can hold.
    total = sum(crops.values()) + sum(animals.values())
    overflow = max(0, total - 96)  # keep the four shed-access tiles open
    for crop in reversed(C.CROPS):
        reduction = min(overflow, crops[crop])
        crops[crop] -= reduction
        overflow -= reduction
    return OpeningTargets(_HANDS[index], crops, animals)
