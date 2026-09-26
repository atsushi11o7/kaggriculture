"""Self-contained repaired tape used before the first shop unlocks."""

from __future__ import annotations

from copy import deepcopy

from kaggriculture.policy.opening.pre_shop_tape import PRE_SHOP_TAPE

LAST_DAY = 2
_PLANNED_OPS = frozenset(("PLANT", "BUILD_COOP", "BUILD_PASTURE"))
_MOVES = frozenset(("NORTH", "SOUTH", "EAST", "WEST"))
_PLAYER_STATE: dict[int, dict] = {}


def applies(day: int) -> bool:
    """Return whether the pre-shop opening owns this day."""
    return day <= LAST_DAY


def _step(observation: dict) -> int:
    return int(observation.get("step", observation["day"] * 24 + observation["hour"]))


def _tile(tiles, position):
    try:
        return tiles[int(position[1])][int(position[0])]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _is_weed(tile) -> bool:
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def _certain_noop(command, tile) -> bool:
    if not command or command[0] == "PASS":
        return True
    op = command[0]
    if op == "PLANT":
        return tile is not None
    if op in ("BUILD_COOP", "BUILD_PASTURE"):
        return tile is not None
    if op == "WATER":
        return (
            not isinstance(tile, dict)
            or tile.get("kind") != "PLANT"
            or bool(tile.get("watered_today"))
        )
    return False


def _state(player: int, step: int) -> dict:
    state = _PLAYER_STATE.get(player)
    if state is None or step == 0 or step <= state["last_step"]:
        state = {"last_step": -1, "pending": {}}
        _PLAYER_STATE[player] = state
    state["last_step"] = step
    return state


def _align_hands(action: dict, count: int) -> None:
    hands = list(action.get("hands") or [])
    hands.extend([["PASS"] for _ in range(max(0, count - len(hands)))])
    action["hands"] = hands[:count]


def _repair_weeds(action: dict, observation: dict, state: dict, step: int) -> None:
    player = int(observation["player"])
    farm = observation["farms"][player]
    positions = [farm["farmer"], *farm["hands"]]
    units = [action.get("farmer") or ["PASS"], *action.get("hands", [])]
    next_action = PRE_SHOP_TAPE[step + 1] if step + 1 < len(PRE_SHOP_TAPE) else {}
    next_units = [next_action.get("farmer") or ["PASS"], *next_action.get("hands", [])]
    pending = state["pending"]

    for index, (position, command) in enumerate(zip(positions, units, strict=True)):
        position = tuple(map(int, position))
        command = list(command)
        tile = _tile(farm["tiles"], position)
        queue = pending.get(index)
        if queue and queue[0][0] != position:
            pending.pop(index, None)
            queue = None

        next_op = next_units[index][0] if index < len(next_units) and next_units[index] else "PASS"
        if command[0] in _PLANNED_OPS and _is_weed(tile):
            pending.setdefault(index, []).append((position, command))
            command = ["DIG"]
        elif queue and _certain_noop(command, tile):
            _, replay = queue[0]
            if replay[0] == "PLANT" and next_op in _MOVES:
                pending.pop(index, None)
            else:
                queue.pop(0)
                if command[0] not in ("PASS", *_MOVES):
                    queue.append((position, command))
                command = replay
                if not queue:
                    pending.pop(index, None)
        elif _is_weed(tile) and _certain_noop(command, tile):
            command = ["DIG"]
        units[index] = command

    action["farmer"] = units[0]
    action["hands"] = units[1:]


def action(observation: dict, configuration: dict | None = None) -> dict:
    """Return the common prefix with physical hand and weed repairs."""
    del configuration
    step = _step(observation)
    if not 0 <= step < len(PRE_SHOP_TAPE):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    result = deepcopy(PRE_SHOP_TAPE[step])
    player = int(observation["player"])
    farm = observation["farms"][player]
    _align_hands(result, len(farm["hands"]))
    _repair_weeds(result, observation, _state(player, step), step)
    return result
