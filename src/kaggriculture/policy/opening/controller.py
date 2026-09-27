"""Route the fixed pre-shop tape and the shop-keyed post-shop tape."""

from __future__ import annotations

from kaggriculture.policy.opening import post_shop_tape, pre_shop

# Kaggle observations are zero-based: replay analysis remains consistent through raw day 11.
OPENING_LAST_OBSERVATION_DAY = 11


def is_opening_day(day: int) -> bool:
    """Return whether a raw Kaggle observation day belongs to the opening."""
    return day <= OPENING_LAST_OBSERVATION_DAY


def opening_action(observation: dict, configuration: dict | None = None) -> dict:
    """Return one opening action selected by the raw observation day.

    Observation days 0 through 2 replay a shop-independent fixed tape.
    Observation days 3 through 11 replay a real top-player tape: a shared
    bridge tape through day 5, then one of 64 tapes selected by the
    (day3, day6) shop pair from day 6 onward.
    """
    if pre_shop.applies(observation["day"]):
        return pre_shop.action(observation, configuration)
    return post_shop_tape.action(observation, configuration)
