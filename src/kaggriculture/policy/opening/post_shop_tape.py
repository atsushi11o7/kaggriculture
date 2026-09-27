"""Shop-keyed tape replay for observation days 3 through 11.

Days 3-5 replay a single shop-independent bridge tape (the day 6 shop isn't
known yet). Day 6 onward locks into one of 64 tapes selected by the
(day3, day6) shop pair -- matching hybrid2965's own shops[:2] router
granularity -- and stays on that tape for the rest of the opening, the same
way hybrid2965 locks in at step 144. Hands reset to empty at every day
boundary regardless of which tape is active, so locking in exactly at a day
boundary means there is no stale hand-index state to reconcile between the
bridge and the newly selected tape.
"""

from __future__ import annotations

from copy import deepcopy

from kaggriculture.policy.opening import land_rule
from kaggriculture.policy.opening import tape_repair as repair
from kaggriculture.policy.opening.bridge_tape import BRIDGE_TAPE
from kaggriculture.policy.opening.tape_bank import lookup
from kaggriculture.rules import constants as C

FIRST_DAY = 3
LAST_DAY = 11
_LOCK_DAY = 6
_LOCK_STEP = _LOCK_DAY * 24
_BRIDGE_START_STEP = FIRST_DAY * 24


class _PlayerState:
    def __init__(self) -> None:
        self.last_step = -1
        self.tape: tuple[dict, ...] | None = None
        self.repair = repair.RepairState()


_PLAYER_STATE: dict[int, _PlayerState] = {}


def applies(day: int) -> bool:
    """Return whether the post-shop tape phase owns this day."""
    return FIRST_DAY <= day <= LAST_DAY


def _step(observation: dict) -> int:
    return int(observation.get("step", observation["day"] * 24 + observation["hour"]))


def _state(player: int, step: int) -> _PlayerState:
    state = _PLAYER_STATE.get(player)
    if state is None or step == 0 or step <= state.last_step:
        state = _PlayerState()
        _PLAYER_STATE[player] = state
    state.last_step = step
    state.repair.reset_if_new_episode(step)
    return state


def _tape_for(observation: dict, state: _PlayerState, step: int) -> tuple[dict, ...]:
    if step < _LOCK_STEP:
        return BRIDGE_TAPE
    if state.tape is None:
        shops = list(observation.get("town", {}).get("unlocked_shops", []) or [])
        day3_shop = shops[0] if shops else ""
        day6_shop = shops[1] if len(shops) > 1 else ""
        state.tape = lookup(day3_shop, day6_shop)
    return state.tape


def _entry(tape: tuple[dict, ...], step: int, phase_start: int) -> dict:
    index = step - phase_start
    if 0 <= index < len(tape):
        return tape[index]
    return {"farmer": ["PASS"], "hands": [], "market": []}


def _ensure_land_purchase(result: dict, observation: dict) -> None:
    """Force BUY_LAND once affordable, even if the tape's own one-shot
    attempt already failed on a different turn.

    The tape's own BUY_LAND timing assumes its source game's cash
    trajectory. The bridge and the locked-in tape come from different real
    games, so cash at the lock boundary can diverge enough that the tape's
    single scripted attempt is unaffordable -- and a literal tape never
    retries. This mirrors hybrid2965's own pattern of layering a small
    reactive fix on top of an otherwise-frozen tape.
    """
    if ["BUY_LAND"] in result["market"]:
        return
    if land_rule.next_forced_purchase(observation) is None:
        return
    market = result["market"]
    if len(market) >= C.MAX_MARKET_ORDERS:
        market.pop()
    market.append(["BUY_LAND"])


def action(observation: dict, configuration: dict | None = None) -> dict:
    """Return one shop-keyed tape action, repaired against the live state."""
    del configuration
    step = _step(observation)
    player = int(observation["player"])
    state = _state(player, step)

    tape = _tape_for(observation, state, step)
    phase_start = _BRIDGE_START_STEP if step < _LOCK_STEP else _LOCK_STEP
    result = deepcopy(_entry(tape, step, phase_start))

    farm = observation["farms"][player]
    repair.align_hands(result, len(farm["hands"]))
    next_action = _entry(tape, step + 1, phase_start)
    repair.repair_weeds(result, observation, state.repair, next_action)
    repair.ensure_feeding(result, observation)
    _ensure_land_purchase(result, observation)
    return result
