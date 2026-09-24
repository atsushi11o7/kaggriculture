"""Resolve fixed-slot Torch intents against sequential simulator constraints."""

from kaggriculture.policy.torch import actions as A
from kaggriculture.policy.torch import candidates as K
from kaggriculture.policy.torch import decode as D
from kaggriculture.rules import constants as C


def copy_environment(obs: dict) -> tuple[dict, dict, dict, dict, list[dict]]:
    """Copy the mutable observation fields used for sequential resolution."""
    player = obs["player"]
    source = obs["farms"][player]
    farm = {
        **source,
        "tiles": [
            [dict(cell) if isinstance(cell, dict) else cell for cell in row]
            for row in source["tiles"]
        ],
    }
    private = obs["private"]
    market = {
        "inventory": dict(obs["market"]["inventory"]),
        "prices": dict(obs["market"]["prices"]),
    }
    return (
        farm,
        dict(private["shed"]),
        dict(private["seeds"]),
        market,
        [dict(inventory) for inventory in private["inventories"]],
    )


def _entry(op: int, item: str | None, quantity: int, *, market: bool) -> list:
    name = C.MARKET_OP_NAMES[op] if market else C.FARMER_OP_NAMES[op]
    result = [name]
    if item is not None:
        result.append(item)
        if market or quantity != 1:
            result.append(quantity)
    return result


def unit_quantity_upper_bound(
    op: int,
    arg: int,
    position: tuple[int, int],
    inventory: dict,
    farm: dict,
) -> int:
    """Return the Torch scoring bound corresponding to the JAX unit mask."""
    item = K.item_name(op, arg, market=False)
    tile = farm["tiles"][position[1]][position[0]]
    if op == C.FARMER_OP_PLACE and not A.is_animal_placement(item, tile):
        return max(inventory.get(item, 0), 1)
    return len(K.QUANTITY_VECTORS)


def execute_units(
    selected_units: list[int],
    selected_quantities: list[int],
    positions: list[tuple[int, int]],
    inventories: list[dict],
    farm: dict,
    shed: dict,
    seeds: dict,
    day: int,
    turns_per_day: int,
    shed_capacity: int,
) -> list[list]:
    """Resolve selected unit intents in simulator order."""
    unit_ops = [K.UNIT_META[choice] for choice in selected_units[: len(positions)]]
    plant_counts = {crop: 0 for crop in C.CROPS}
    for op, arg in unit_ops:
        if op == C.FARMER_OP_PLANT:
            plant_counts[C.CROPS[arg]] += 1
    blocked_crops = {crop for crop, count in plant_counts.items() if count > seeds.get(crop, 0)}

    entries = []
    for slot, ((op, arg), position, inventory) in enumerate(
        zip(unit_ops, positions, inventories, strict=True)
    ):
        item = K.item_name(op, arg, market=False)
        if op == C.FARMER_OP_PLANT and item in blocked_crops:
            entries.append(["PASS"])
            continue
        op_name = C.FARMER_OP_NAMES[op]
        tile = farm["tiles"][position[1]][position[0]]
        quantity = 1
        if A.requires_quantity(op_name, item, tile):
            maximum = A.max_executable_quantity(
                op_name, item, farm, shed, None, shed_capacity, inventory=inventory
            )
            quantity = max(min(selected_quantities[slot], maximum), 1)
        entries.append(_entry(op, item, quantity, market=False))
        D.commit_unit_action(
            farm,
            shed,
            seeds,
            op_name,
            item,
            position,
            day,
            n=quantity,
            shed_capacity=shed_capacity,
            inventory=inventory,
            turns_per_day=turns_per_day,
        )
    return entries


def execute_market(
    selected_market: list[int],
    selected_quantities: list[int],
    farm: dict,
    shed: dict,
    market: dict,
    *,
    shed_capacity: int,
    hire_mult: float,
) -> list[list]:
    """Resolve market intents until STOP while updating the market shadow state."""
    entries: list[list] = []
    for offset, choice in enumerate(selected_market):
        op, arg = K.MARKET_META[choice]
        if op == C.N_MARKET_OPS + 1:
            break
        if op == C.N_MARKET_OPS:
            entries.append(list(A.MARKET_WAIT_ACTION))
            continue
        item = K.item_name(op, arg, market=True)
        legal = {
            K.key(vector)
            for vector in A.legal_market_actions(farm, shed, market, hire_mult, shed_capacity)
        }
        if K.key(K.MARKET_VECTORS[choice]) not in legal:
            entries.append(list(A.MARKET_WAIT_ACTION))
            continue
        op_name = C.MARKET_OP_NAMES[op]
        quantity = 1
        if A.requires_quantity(op_name, item):
            maximum = A.max_executable_quantity(op_name, item, farm, shed, market, shed_capacity)
            quantity = min(selected_quantities[offset], maximum)
            if quantity <= 0:
                entries.append(list(A.MARKET_WAIT_ACTION))
                continue
        entries.append(_entry(op, item, quantity, market=True))
        D.commit_market_action(
            farm, shed, market, op_name, item, quantity, shed_capacity, hire_mult
        )
    return entries


def resolve_action(
    obs: dict,
    selected_units: list[int],
    selected_market: list[int],
    selected_quantities: list[int],
    *,
    turns_per_day: int,
    shed_capacity: int,
    hire_mult: float,
) -> dict:
    """Convert fixed-slot choices to one simulator-compatible action dictionary."""
    player = obs["player"]
    farm_view = obs["farms"][player]
    positions = [farm_view["farmer"], *farm_view["hands"]]
    farm, shed, seeds, market, inventories = copy_environment(obs)
    units = execute_units(
        selected_units,
        selected_quantities,
        positions,
        inventories,
        farm,
        shed,
        seeds,
        obs["day"],
        turns_per_day,
        shed_capacity,
    )
    markets = execute_market(
        selected_market,
        selected_quantities[C.MAX_HANDS + 1 :],
        farm,
        shed,
        market,
        shed_capacity=shed_capacity,
        hire_mult=hire_mult,
    )
    return {"farmer": units[0], "hands": units[1:], "market": markets}
