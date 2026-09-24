"""Backend-independent endgame deadlines."""

from kaggriculture.rules import constants as C
from kaggriculture.rules import game_params as P


def last_action_day(episode_steps: int, turns_per_day: int) -> int:
    """Return the day containing the final actionable step."""
    return (episode_steps - 2) // turns_per_day


def investment_allowed(
    op: int,
    arg: int,
    *,
    market: bool,
    day: int,
    turns_per_day: int = 24,
    episode_steps: int = 720,
) -> bool:
    """Return whether a new biological investment can produce before termination."""
    final_day = last_action_day(episode_steps, turns_per_day)
    if not market and op == C.FARMER_OP_PLANT:
        return day + P.CROP_FIRST_YIELD_DAY[arg] < final_day
    if market and op == C.MARKET_OP_BUY_SEED:
        return day + P.CROP_FIRST_YIELD_DAY[arg] < final_day
    if market and op == C.MARKET_OP_BUY_ANIMAL:
        return day + P.ANIMAL_FIRST_YIELD_DAY[arg] < final_day
    if market and op == C.MARKET_OP_BUY_LAND:
        return day + min(P.CROP_FIRST_YIELD_DAY) < final_day
    return True
