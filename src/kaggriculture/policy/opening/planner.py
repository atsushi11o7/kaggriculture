"""State-driven task and market planner for human-readable days 4 through 11."""

from __future__ import annotations

from dataclasses import dataclass

from kaggriculture.policy.opening.land_rule import next_forced_purchase, required_capital
from kaggriculture.policy.opening.targets import OpeningTargets, for_state
from kaggriculture.rules import constants as C
from kaggriculture.rules import game_params as P

_SHED_TILES = frozenset(((4, 4), (5, 4), (4, 5), (5, 5)))
_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
_FIB = (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987)
_CROP_INVESTMENT_ORDER = ("STRAWBERRY", "WHEAT", "TOMATO", "CARROT", "MELON")
_OPERATING_CASH_FLOOR = 100
_EXPANSION_CASH_RESERVE = 300
_LAND_ACTIVATION_RESERVE = 220
_SE_INITIAL_TILES = 8


@dataclass(frozen=True)
class Task:
    """One exclusive physical job assignable to a unit."""

    priority: int
    position: tuple[int, int]
    command: tuple[str, ...]
    required_item: str | None = None


# Per-seat, within-day route commitments. They prevent workers from exchanging
# distant jobs every turn while still allowing completed jobs to disappear.
_ROUTES: dict[int, tuple[int, int, dict[int, Task]]] = {}


def _same_job(left: Task, right: Task) -> bool:
    return (left.position, left.command, left.required_item) == (
        right.position,
        right.command,
        right.required_item,
    )


def _distance(left, right) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _move(position, target) -> list[str]:
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


def _counts(tiles) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    crops = dict.fromkeys(C.CROPS, 0)
    animals = dict.fromkeys(C.ANIMALS, 0)
    structures = {"COOP": 0, "PASTURE": 0}
    for row in tiles:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT" and tile.get("crop") in crops:
                crops[tile["crop"]] += 1
            if tile.get("kind") in structures:
                structures[tile["kind"]] += 1
                if tile.get("animal") in animals:
                    animals[tile["animal"]] += 1
    return crops, animals, structures


def _held(private: dict, item: str) -> int:
    return int(private["shed"].get(item, 0)) + sum(
        int(inventory.get(item, 0)) for inventory in private["inventories"]
    )


def _quadrant(position: tuple[int, int]) -> str:
    x, y = position
    return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")


def _tile_positions(tiles, predicate) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y, row in enumerate(tiles)
        for x, tile in enumerate(row)
        if (x, y) not in _SHED_TILES and predicate(tile)
    ]


def _spread_positions(tiles, positions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Order candidate tiles so unlocked quadrants develop evenly."""
    occupancy = dict.fromkeys(C.QUADRANTS, 0)
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and (x, y) not in _SHED_TILES:
                occupancy[_quadrant((x, y))] += 1

    remaining = {quadrant: [] for quadrant in C.QUADRANTS}
    for position in positions:
        remaining[_quadrant(position)].append(position)
    for candidates in remaining.values():
        candidates.sort(
            key=lambda position: (_distance(position, _nearest_shed(position)), position)
        )

    result: list[tuple[int, int]] = []
    while any(remaining.values()):
        available = [quadrant for quadrant, candidates in remaining.items() if candidates]
        quadrant = min(available, key=lambda name: (occupancy[name], C.QUADRANTS.index(name)))
        result.append(remaining[quadrant].pop(0))
        occupancy[quadrant] += 1
    return result


def _mature(tile: dict, day: int) -> bool:
    if tile.get("kind") != "PLANT" or tile.get("crop") not in C.CROPS:
        return True
    crop = C.CROPS.index(tile["crop"])
    return day - int(tile.get("planted_day", day)) >= P.CROP_FIRST_YIELD_DAY[crop]


def _tasks(observation: dict, targets: OpeningTargets) -> list[Task]:
    player = observation["player"]
    farm = observation["farms"][player]
    private = observation["private"]
    tiles = farm["tiles"]
    day = observation["day"]
    crop_count, animal_count, structures = _counts(tiles)
    tasks: list[Task] = []

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            position = (x, y)
            kind = tile.get("kind")
            if kind == "PLANT":
                if not tile.get("watered_today", False):
                    urgency = 0 if int(tile.get("consecutive_unwatered", 0)) >= 1 else 10
                    tasks.append(Task(urgency, position, ("WATER",)))
                if int(tile.get("yield_units", 0)) > 0 and _mature(tile, day):
                    tasks.append(Task(20, position, ("HARVEST",)))
            animal = tile.get("animal")
            if animal in C.ANIMALS:
                if not tile.get("fed_today", False):
                    urgency = 0 if int(tile.get("consecutive_unfed", 0)) >= 1 else 10
                    tasks.append(Task(urgency, position, ("FEED",), "WHEAT"))
                if int(tile.get("yield_units", 0)) > 0:
                    tasks.append(Task(20, position, ("HARVEST",)))
                if tile.get("fertilizer_available", False):
                    tasks.append(Task(24, position, ("COLLECT_FERTILIZER",)))
                if not tile.get("cared_today", False):
                    tasks.append(Task(55, position, ("CARE",)))

    # Place already purchased animals before creating more structures or crops.
    # Fill the shed-closest empty structure first to keep daily feed/care
    # travel short.
    empty_structures: dict[str, list[tuple[int, int]]] = {"COOP": [], "PASTURE": []}
    for kind in empty_structures:
        positions = _tile_positions(
            tiles,
            lambda tile, kind=kind: (
                isinstance(tile, dict) and tile.get("kind") == kind and tile.get("animal") is None
            ),
        )
        positions.sort(key=lambda position: _distance(position, _nearest_shed(position)))
        empty_structures[kind] = positions
    for animal in C.ANIMALS:
        available = _held(private, animal)
        positions = empty_structures[_STRUCTURE[animal]]
        for position in positions[:available]:
            tasks.append(Task(15, position, ("PLACE", animal), animal))
        del positions[:available]

    empty = _spread_positions(tiles, _tile_positions(tiles, lambda tile: tile is None))
    weeds = _spread_positions(
        tiles,
        _tile_positions(tiles, lambda tile: isinstance(tile, dict) and tile.get("kind") == "WEED"),
    )
    reserved: set[tuple[int, int]] = set()

    # Build only for animals already owned. Constructing the full strategic
    # target before those animals are affordable wastes both tiles and labour.
    needed_structures = {
        "COOP": max(0, animal_count["GOOSE"] + _held(private, "GOOSE") - structures["COOP"]),
        "PASTURE": max(
            0,
            animal_count["COW"]
            + animal_count["SHEEP"]
            + _held(private, "COW")
            + _held(private, "SHEEP")
            - structures["PASTURE"],
        ),
    }

    # Reserve tiles for animals already bought before crops claim the rest.
    # needed_structures only counts animals already owned (never the full
    # strategic target), so this is a small, bounded claim that stops
    # purchased livestock from sitting unplaced in the shed indefinitely
    # while large crop targets exhaust every empty tile first.
    for kind, command in (("COOP", "BUILD_COOP"), ("PASTURE", "BUILD_PASTURE")):
        for position in empty:
            if not needed_structures[kind] or position in reserved:
                continue
            tasks.append(Task(40, position, (command,)))
            reserved.add(position)
            needed_structures[kind] -= 1

    se_occupied = sum(
        isinstance(tile, dict)
        for y, row in enumerate(tiles)
        for x, tile in enumerate(row)
        if _quadrant((x, y)) == "SE"
    )
    se_activation_remaining = max(0, _SE_INITIAL_TILES - se_occupied) if 10 <= day <= 11 else 0
    seed_stock = private["seeds"]
    for crop in _CROP_INVESTMENT_ORDER:
        needed = max(0, targets.crops[crop] - crop_count[crop])
        available = min(needed, int(seed_stock.get(crop, 0)))
        for position in empty:
            if not available or position in reserved:
                continue
            priority = 30
            if _quadrant(position) == "SE" and se_activation_remaining:
                priority = 8
                se_activation_remaining -= 1
            tasks.append(Task(priority, position, ("PLANT", crop)))
            reserved.add(position)
            available -= 1

    unfilled = sum(needed_structures.values()) + sum(
        max(0, targets.crops[crop] - crop_count[crop] - int(seed_stock.get(crop, 0)))
        for crop in C.CROPS
    )
    for position in weeds[:unfilled]:
        tasks.append(Task(35, position, ("DIG",)))

    # Python's sort is stable: keep the quadrant-spreading order within each
    # priority instead of collapsing it back to NW-first coordinate order.
    return sorted(tasks, key=lambda task: task.priority)


def _allocate_budget(budget: float, shortfalls: dict[str, tuple[int, int]]) -> dict[str, int]:
    """Split a cash budget across items by dollar-value shortfall.

    Buying items in a fixed priority order lets earlier, cheaper items
    exhaust a scarce budget and starve everything after them (this is how
    SHEEP, the costliest animal and last in iteration order, could go an
    entire game without a single purchase). Each item instead gets a share
    of the budget proportional to its own remaining shortfall value, then
    buys as many units as that share affords.
    """
    total_value = sum(cost * missing for cost, missing in shortfalls.values())
    if total_value <= 0:
        return {}
    quantities: dict[str, int] = {}
    for name, (cost, missing) in shortfalls.items():
        if missing <= 0 or cost <= 0:
            continue
        share = budget * (cost * missing) / total_value
        quantity = min(missing, int(share // cost))
        if quantity:
            quantities[name] = quantity
    return quantities


def _job_pending(task: Task, observation: dict) -> bool:
    """Return whether a committed physical job can still be completed."""
    player = observation["player"]
    x, y = task.position
    tile = observation["farms"][player]["tiles"][y][x]
    op = task.command[0]
    if op == "PLANT":
        crop = task.command[1]
        return tile is None and int(observation["private"]["seeds"].get(crop, 0)) > 0
    if op in {"BUILD_COOP", "BUILD_PASTURE"}:
        return tile is None
    return False


def _unit_commands(observation: dict, tasks: list[Task]) -> list[list[str]]:
    player = observation["player"]
    farm = observation["farms"][player]
    private = observation["private"]
    positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
    inventories = private["inventories"]
    commands: list[list[str] | None] = [None] * len(positions)
    available_units = set(range(len(positions)))

    step = int(observation.get("step", 0))
    land_count = len(farm["unlocked_quadrants"])
    previous_step, previous_land_count, routes = _ROUTES.get(player, (-1, land_count, {}))
    if (
        step <= previous_step
        or int(observation.get("hour", 0)) == 0
        or land_count != previous_land_count
    ):
        routes = {}
    else:
        routes = {unit: task for unit, task in routes.items() if unit < len(positions)}

    required_items = {task.required_item for task in tasks if task.required_item}
    # Return sale cargo before assigning new work. Keep items needed by an
    # outstanding task so a WHEAT carrier can feed an animal directly.
    for unit in tuple(available_units):
        inventory = inventories[unit] if unit < len(inventories) else {}
        sale_cargo = any(
            int(inventory.get(item, 0)) > 0 and item not in required_items for item in C.PRODUCTS
        )
        if not sale_cargo:
            continue
        target = _nearest_shed(positions[unit])
        commands[unit] = ["DROP"] if positions[unit] == target else _move(positions[unit], target)
        available_units.remove(unit)
        routes.pop(unit, None)

    remaining_tasks = list(tasks)
    for unit, previous in tuple(routes.items()):
        if unit not in available_units:
            continue
        match = next((task for task in remaining_tasks if _same_job(task, previous)), None)
        if match is None and _job_pending(previous, observation):
            match = previous
        if match is None:
            routes.pop(unit, None)
            continue
        position = positions[unit]
        inventory = inventories[unit] if unit < len(inventories) else {}
        if match.required_item and int(inventory.get(match.required_item, 0)) <= 0:
            shed_target = _nearest_shed(position)
            if position == shed_target and int(private["shed"].get(match.required_item, 0)) > 0:
                commands[unit] = ["PICKUP", match.required_item, 1]
            else:
                commands[unit] = _move(position, shed_target)
        elif position == match.position:
            commands[unit] = list(match.command)
        else:
            commands[unit] = _move(position, match.position)
        routes[unit] = match
        if match in remaining_tasks:
            remaining_tasks.remove(match)
        available_units.remove(unit)

    shed = private["shed"]
    for task in remaining_tasks:
        if not available_units:
            break

        def cost(unit: int, task: Task = task) -> tuple[int, int]:
            inventory = inventories[unit] if unit < len(inventories) else {}
            if task.required_item and int(inventory.get(task.required_item, 0)) <= 0:
                via_shed = _distance(positions[unit], _nearest_shed(positions[unit]))
                via_shed += min(_distance(shed_pos, task.position) for shed_pos in _SHED_TILES)
                return via_shed + 1, unit
            return _distance(positions[unit], task.position), unit

        unit = min(available_units, key=cost)
        position = positions[unit]
        inventory = inventories[unit] if unit < len(inventories) else {}
        if task.required_item and int(inventory.get(task.required_item, 0)) <= 0:
            shed_target = _nearest_shed(position)
            if position == shed_target and int(shed.get(task.required_item, 0)) > 0:
                commands[unit] = ["PICKUP", task.required_item, 1]
            else:
                commands[unit] = _move(position, shed_target)
        elif position == task.position:
            commands[unit] = list(task.command)
        else:
            commands[unit] = _move(position, task.position)
        routes[unit] = task
        available_units.remove(unit)

    _ROUTES[player] = (step, land_count, routes)
    return [command or ["PASS"] for command in commands]


def _market_orders(observation: dict, targets: OpeningTargets) -> list[list]:
    player = observation["player"]
    farm = observation["farms"][player]
    private = observation["private"]
    crop_count, animal_count, _ = _counts(farm["tiles"])
    cash = float(farm["money"])
    settled_cash = cash
    orders: list[list] = []

    prices = observation["market"]["prices"]
    animal_total = sum(animal_count.values())
    wheat_reserve = animal_total + 2
    sale_stock = {
        product: max(
            0,
            int(private["shed"].get(product, 0)) - (wheat_reserve if product == "WHEAT" else 0),
        )
        for product in C.PRODUCTS
    }
    # Convert completed production into opening capital. Limiting this to the
    # three most valuable lots leaves room for land, labour and input orders.
    for product in sorted(
        C.PRODUCTS,
        key=lambda item: (sale_stock[item] * int(prices.get(item, 1)), item),
        reverse=True,
    )[:3]:
        quantity = sale_stock[product]
        if quantity:
            orders.append(["SELL", product, quantity])
            cash += quantity * int(prices.get(product, 1))

    land = next_forced_purchase(observation, cash)
    planned_land_purchases = int(land is not None)
    if land is not None:
        orders.append(land)
        land_cost = P.LAND_PRICES[len(farm["unlocked_quadrants"]) - 1]
        cash -= land_cost
        settled_cash -= land_cost

    land_count_after_orders = len(farm["unlocked_quadrants"]) + planned_land_purchases
    final_opening_day = observation["day"] == 10 and land_count_after_orders == len(C.QUADRANTS)
    hire_cash_floor = 0 if final_opening_day else _OPERATING_CASH_FLOOR

    missing_hands = max(0, targets.hands - len(farm["hands"]))
    hires_today = int(farm.get("hires_today", 0))
    for offset in range(min(missing_hands, 3)):
        index = min(hires_today + offset, len(_FIB) - 1)
        cost = _FIB[index]
        if cash - cost < hire_cash_floor or len(orders) >= C.MAX_MARKET_ORDERS:
            break
        orders.append(["HIRE"])
        cash -= cost
        settled_cash -= cost

    if planned_land_purchases and land_count_after_orders == len(C.QUADRANTS):
        wheat_seed_cost = P.CROP_SEED_COST[C.CROPS.index("WHEAT")]
        activation_seeds = min(
            _SE_INITIAL_TILES,
            int(max(0, cash - _LAND_ACTIVATION_RESERVE) // wheat_seed_cost),
        )
        if activation_seeds and len(orders) < C.MAX_MARKET_ORDERS:
            orders.append(["BUY_SEED", "WHEAT", activation_seeds])
            cash -= activation_seeds * wheat_seed_cost
            settled_cash -= activation_seeds * wheat_seed_cost

    wheat = _held(private, "WHEAT")
    feed_shortfall = max(0, animal_total + 2 - wheat)
    if feed_shortfall and len(orders) < C.MAX_MARKET_ORDERS:
        price = max(1, int(prices.get("WHEAT", 25)))
        quantity = min(feed_shortfall, int(max(0, cash - _OPERATING_CASH_FLOOR) // price))
        if quantity:
            orders.append(["BUY_PRODUCT", "WHEAT", quantity])
            cash -= quantity * price
            settled_cash -= quantity * price

    # Expansion inputs use only cash above the next land reserve. Daily labour
    # and feed remain funded because they maintain the assets already owned.
    expansion_reserve = max(
        _EXPANSION_CASH_RESERVE,
        required_capital(observation, planned_land_purchases),
    )
    # Same-turn bulk sales change their own execution price and also interact
    # with the opponent's orders. Optional expansion therefore uses only cash
    # settled before this turn; realized sale proceeds become available from
    # the next observation.
    expansion_cash = max(0.0, min(cash, settled_cash) - expansion_reserve)

    seed_cost = dict(zip(C.CROPS, P.CROP_SEED_COST, strict=True))
    animal_cost = dict(zip(C.ANIMALS, P.ANIMAL_COST, strict=True))
    crop_shortfall_value = sum(
        max(0, targets.crops[crop] - crop_count[crop] - int(private["seeds"].get(crop, 0)))
        * seed_cost[crop]
        for crop in C.CROPS
    )
    animal_shortfall_value = sum(
        max(0, targets.animals[animal] - animal_count[animal] - _held(private, animal))
        * animal_cost[animal]
        for animal in C.ANIMALS
    )
    # Split expansion cash between crops and animals by their remaining
    # dollar-value shortfall. Spending crops-first every turn otherwise
    # starves animal purchases, since crop targets need far more tiles.
    total_shortfall_value = crop_shortfall_value + animal_shortfall_value
    if total_shortfall_value > 0:
        animal_cash = expansion_cash * animal_shortfall_value / total_shortfall_value
    else:
        animal_cash = 0.0
    crop_cash = expansion_cash - animal_cash

    crop_purchases = _allocate_budget(
        crop_cash,
        {
            crop: (
                seed_cost[crop],
                max(0, targets.crops[crop] - crop_count[crop] - int(private["seeds"].get(crop, 0))),
            )
            for crop in _CROP_INVESTMENT_ORDER
        },
    )
    for crop in _CROP_INVESTMENT_ORDER:
        quantity = crop_purchases.get(crop, 0)
        if quantity and len(orders) < C.MAX_MARKET_ORDERS:
            orders.append(["BUY_SEED", crop, quantity])
            cash -= quantity * seed_cost[crop]

    animal_purchases = _allocate_budget(
        animal_cash,
        {
            animal: (
                animal_cost[animal],
                max(0, targets.animals[animal] - animal_count[animal] - _held(private, animal)),
            )
            for animal in C.ANIMALS
        },
    )
    for animal in C.ANIMALS:
        quantity = animal_purchases.get(animal, 0)
        if quantity and len(orders) < C.MAX_MARKET_ORDERS:
            orders.append(["BUY_ANIMAL", animal, quantity])
            cash -= quantity * animal_cost[animal]

    return orders[: C.MAX_MARKET_ORDERS]


def action(observation: dict, configuration: dict | None = None) -> dict:
    """Plan one shop-aware opening turn from the current physical state."""
    del configuration
    shops = list(observation.get("town", {}).get("unlocked_shops", []) or [])
    targets = for_state(observation["day"], shops)
    tasks = _tasks(observation, targets)
    commands = _unit_commands(observation, tasks)
    return {
        "farmer": commands[0],
        "hands": commands[1:],
        "market": _market_orders(observation, targets),
    }
