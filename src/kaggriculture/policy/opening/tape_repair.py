"""Reactive repair shared by every tape-replay phase of the opening.

A frozen action tape assumes a specific hand count and tile layout at each
step. The live game rarely matches exactly (hiring can lag a turn, and a
weed can appear where the tape assumed an empty or already-worked tile).
This module keeps the tape usable despite that drift: it pads or truncates
hand commands to the real hired count, and detects a weed blocking a
planned PLANT/BUILD action and detours through DIG before replaying the
original command on a later turn.
"""

from __future__ import annotations

_PLANNED_OPS = frozenset(("PLANT", "BUILD_COOP", "BUILD_PASTURE"))
_MOVES = frozenset(("NORTH", "SOUTH", "EAST", "WEST"))
_PREEMPTABLE = frozenset({"PASS", *_MOVES})
_SHED_TILES = frozenset(((4, 4), (5, 4), (4, 5), (5, 5)))


def _tile(tiles, position):
    try:
        return tiles[int(position[1])][int(position[0])]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _distance(left, right) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _move_toward(position, target) -> list[str]:
    x, y = position
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def _nearest_shed(position) -> tuple[int, int]:
    return min(_SHED_TILES, key=lambda target: (_distance(position, target), target))


def is_weed(tile) -> bool:
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


def ensure_feeding(action: dict, observation: dict) -> None:
    """Route a unit to feed any animal that will escape today.

    An animal escapes (is permanently lost) after 2 consecutive unfed days.
    A tape's FEED commands are pinned to specific unit indices, so any drift
    between the tape's assumed hand layout and the live one (fewer hands
    hired than the source game had, a weed detour shifting a unit's
    schedule) can silently drop a whole hand's duties -- with no crash to
    signal it, just a dead animal a day later. This preempts any unit that
    is only passing or moving (never a unit already doing real work) and
    walks it to the shed for wheat if needed, then to the animal.
    """
    player = int(observation["player"])
    farm = observation["farms"][player]
    private = observation["private"]
    positions = [tuple(map(int, farm["farmer"])), *(tuple(map(int, h)) for h in farm["hands"])]
    inventories = private["inventories"]
    shed = private["shed"]
    units = [action.get("farmer") or ["PASS"], *action.get("hands", [])]

    urgent_tiles = [
        (x, y)
        for y, row in enumerate(farm["tiles"])
        for x, tile in enumerate(row)
        if isinstance(tile, dict)
        and tile.get("animal") is not None
        and not tile.get("fed_today")
        and int(tile.get("consecutive_unfed", 0)) >= 1
    ]
    if not urgent_tiles:
        return
    already_handled = {
        positions[index] for index, command in enumerate(units) if command and command[0] == "FEED"
    }
    urgent_tiles = [tile for tile in urgent_tiles if tile not in already_handled]

    available = [index for index, command in enumerate(units) if command[0] in _PREEMPTABLE]
    for tile_position in urgent_tiles:
        if not available:
            break

        def cost(index: int, tile_position: tuple[int, int] = tile_position) -> tuple[int, int]:
            position = positions[index]
            inventory = inventories[index] if index < len(inventories) else {}
            if int(inventory.get("WHEAT", 0)) > 0:
                return _distance(position, tile_position), index
            via_shed = _distance(position, _nearest_shed(position))
            via_shed += min(_distance(shed_pos, tile_position) for shed_pos in _SHED_TILES)
            return via_shed + 1, index

        index = min(available, key=cost)
        position = positions[index]
        inventory = inventories[index] if index < len(inventories) else {}
        if int(inventory.get("WHEAT", 0)) > 0:
            units[index] = (
                ["FEED"] if position == tile_position else _move_toward(position, tile_position)
            )
        else:
            shed_target = _nearest_shed(position)
            if position == shed_target and int(shed.get("WHEAT", 0)) > 0:
                units[index] = ["PICKUP", "WHEAT", 1]
            else:
                units[index] = _move_toward(position, shed_target)
        available.remove(index)

    action["farmer"] = units[0]
    action["hands"] = units[1:]


def align_hands(action: dict, count: int) -> None:
    """Pad or truncate hand commands to the actually hired hand count."""
    hands = list(action.get("hands") or [])
    hands.extend([["PASS"] for _ in range(max(0, count - len(hands)))])
    action["hands"] = hands[:count]


class RepairState:
    """Per-player pending weed-detour queue, tracked across a tape replay."""

    def __init__(self) -> None:
        self.last_step = -1
        self.pending: dict[int, list] = {}

    def reset_if_new_episode(self, step: int) -> None:
        if step == 0 or step <= self.last_step:
            self.pending = {}
        self.last_step = step


def repair_weeds(
    action: dict,
    observation: dict,
    state: RepairState,
    next_action: dict | None,
) -> None:
    """Mutate `action` in place, detouring through DIG around live weeds."""
    player = int(observation["player"])
    farm = observation["farms"][player]
    positions = [farm["farmer"], *farm["hands"]]
    units = [action.get("farmer") or ["PASS"], *action.get("hands", [])]
    next_units = (
        [next_action.get("farmer") or ["PASS"], *next_action.get("hands", [])]
        if next_action is not None
        else []
    )
    pending = state.pending

    for index, (position, command) in enumerate(zip(positions, units, strict=True)):
        position = tuple(map(int, position))
        command = list(command)
        tile = _tile(farm["tiles"], position)
        queue = pending.get(index)
        if queue and queue[0][0] != position:
            pending.pop(index, None)
            queue = None

        next_op = next_units[index][0] if index < len(next_units) and next_units[index] else "PASS"
        if command[0] in _PLANNED_OPS and is_weed(tile):
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
        elif is_weed(tile) and _certain_noop(command, tile):
            command = ["DIG"]
        units[index] = command

    action["farmer"] = units[0]
    action["hands"] = units[1:]
