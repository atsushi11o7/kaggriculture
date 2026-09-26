"""Day- and shop-conditioned production targets for the opening."""

from __future__ import annotations

from dataclasses import dataclass

from kaggriculture.policy.opening.demand import product_demand
from kaggriculture.rules import constants as C


@dataclass(frozen=True)
class OpeningTargets:
    """Desired active capacity at the current opening day."""

    hands: int
    crops: dict[str, int]
    animals: dict[str, int]


_HANDS = (6, 6, 6, 8, 9, 9, 10, 11, 11)  # observation days 3..11


def _base_crops(day: int) -> dict[str, int]:
    if day <= 5:
        return {"WHEAT": 10, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 9, "MELON": 11}
    if day <= 8:
        wheat = 12 + (day - 6) * 4
        strawberry = 17 + (day - 6) * 2
        return {
            "WHEAT": wheat,
            "CARROT": 0,
            "TOMATO": 0,
            "STRAWBERRY": strawberry,
            "MELON": 11,
        }
    return {
        "WHEAT": 25 + (day - 9) * 3,
        "CARROT": 2,
        "TOMATO": 2,
        "STRAWBERRY": 23 + (day - 9) * 2,
        "MELON": 0,
    }


def _base_animals(day: int) -> dict[str, int]:
    if day <= 5:
        return {"GOOSE": 0, "COW": 2, "SHEEP": 3}
    if day <= 8:
        return {"GOOSE": 2, "COW": 6, "SHEEP": 4}
    return {"GOOSE": 5, "COW": 9, "SHEEP": 5}


def for_state(day: int, unlocked_shops: list[str]) -> OpeningTargets:
    """Build targets for a raw observation day and unlocked shop demand."""
    index = min(max(day, 3), 11) - 3
    crops = _base_crops(day)
    animals = _base_animals(day)
    demand = product_demand(unlocked_shops)

    yarn_count = unlocked_shops.count("YARN_STORE")
    if yarn_count:
        animals["SHEEP"] += 7 + 3 * (yarn_count - 1)
    animals["COW"] += 2 * demand["MILK"]
    animals["GOOSE"] += 2 * demand["EGG"]

    crops["WHEAT"] += 2 * demand["WHEAT"]
    crops["CARROT"] += 3 * demand["CARROT"]
    crops["TOMATO"] += 2 * demand["TOMATO"]
    crops["STRAWBERRY"] += 2 * demand["STRAWBERRY"]

    # Never request more occupied tiles than the fully unlocked board can hold.
    total = sum(crops.values()) + sum(animals.values())
    overflow = max(0, total - 96)  # keep the four shed-access tiles open
    for crop in reversed(C.CROPS):
        reduction = min(overflow, crops[crop])
        crops[crop] -= reduction
        overflow -= reduction
    return OpeningTargets(_HANDS[index], crops, animals)
