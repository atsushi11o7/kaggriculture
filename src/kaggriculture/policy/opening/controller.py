"""Route the fixed pre-shop tape and the state-driven opening planner."""

from __future__ import annotations

from kaggriculture.policy.opening import planner, pre_shop

# Kaggle observations are zero-based: replay analysis remains consistent through raw day 11.
OPENING_LAST_OBSERVATION_DAY = 11


def is_opening_day(day: int) -> bool:
    """Return whether a raw Kaggle observation day belongs to the opening."""
    return day <= OPENING_LAST_OBSERVATION_DAY


def opening_action(observation: dict, configuration: dict | None = None) -> dict:
    """Return one opening action selected by the raw observation day.

    Observation days 0 through 2 (displayed days 1 through 3) use the fixed
    tape. Observation days 3 through 11 (displayed days 4 through 11) replan
    work and market orders from state.
    """
    if pre_shop.applies(observation["day"]):
        return pre_shop.action(observation, configuration)
    return planner.action(observation, configuration)
