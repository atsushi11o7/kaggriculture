"""CPU endgame action overrides for submission inference."""

from __future__ import annotations

from kaggriculture.rules import constants as C


def _shed_targets(board_size: int) -> tuple[tuple[int, int], ...]:
    left = board_size // 2 - 1
    right = board_size // 2
    return ((left, left), (right, left), (left, right), (right, right))


def _return_command(position, targets) -> list[str]:
    x, y = position
    tx, ty = min(targets, key=lambda target: abs(x - target[0]) + abs(y - target[1]))
    if (x, y) == (tx, ty):
        return ["DROP"]
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    return ["NORTH"]


def apply(
    observation: dict,
    action: dict,
    *,
    episode_steps: int = 720,
    turns_per_day: int = 24,
    board_size: int = 10,
) -> dict:
    """Apply forced cargo return and final shed liquidation to one action."""
    step = observation["day"] * turns_per_day + observation["hour"]
    last_step = episode_steps - 2
    remaining = last_step - step + 1
    if remaining <= 0:
        return action

    player = observation["player"]
    farm = observation["farms"][player]
    positions = [farm["farmer"], *farm["hands"]]
    inventories = observation["private"]["inventories"]
    units = [list(action.get("farmer") or ["PASS"]), *map(list, action.get("hands", []))]
    units.extend([["PASS"]] * (len(positions) - len(units)))
    targets = _shed_targets(board_size)
    for index, (position, inventory) in enumerate(zip(positions, inventories, strict=True)):
        if not any(inventory.get(item, 0) > 0 for item in C.PRODUCTS):
            continue
        distance = min(
            abs(position[0] - target[0]) + abs(position[1] - target[1]) for target in targets
        )
        if distance + 1 >= remaining:
            units[index] = _return_command(position, targets)

    market = action.get("market", [])
    if step == last_step:
        available = dict(observation["private"]["shed"])
        for position, inventory in zip(positions, inventories, strict=True):
            if tuple(position) in targets:
                for product in C.PRODUCTS:
                    available[product] = available.get(product, 0) + inventory.get(product, 0)
        market = [
            ["SELL", product, min(available.get(product, 0), 100)]
            for product in C.PRODUCTS
            if available.get(product, 0) > 0
        ]
    return {"farmer": units[0], "hands": units[1:], "market": market}
