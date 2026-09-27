"""Self-contained repaired tape used before the first shop unlocks."""

from __future__ import annotations

from copy import deepcopy

from kaggriculture.policy.opening import tape_repair as repair
from kaggriculture.policy.opening.pre_shop_tape import PRE_SHOP_TAPE

LAST_DAY = 2
_PLAYER_STATE: dict[int, repair.RepairState] = {}


def applies(day: int) -> bool:
    """Return whether the pre-shop opening owns this day."""
    return day <= LAST_DAY


def _step(observation: dict) -> int:
    return int(observation.get("step", observation["day"] * 24 + observation["hour"]))


def _state(player: int, step: int) -> repair.RepairState:
    state = _PLAYER_STATE.setdefault(player, repair.RepairState())
    state.reset_if_new_episode(step)
    return state


def action(observation: dict, configuration: dict | None = None) -> dict:
    """Return the common prefix with physical hand and weed repairs."""
    del configuration
    step = _step(observation)
    if not 0 <= step < len(PRE_SHOP_TAPE):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    result = deepcopy(PRE_SHOP_TAPE[step])
    player = int(observation["player"])
    farm = observation["farms"][player]
    repair.align_hands(result, len(farm["hands"]))
    next_action = PRE_SHOP_TAPE[step + 1] if step + 1 < len(PRE_SHOP_TAPE) else None
    repair.repair_weeds(result, observation, _state(player, step), next_action)
    repair.ensure_feeding(result, observation)
    return result
