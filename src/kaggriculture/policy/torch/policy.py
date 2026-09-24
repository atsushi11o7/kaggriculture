"""CPU inference for the fixed-slot policy without a JAX dependency."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.torch import actions as A
from kaggriculture.policy.torch import decode as DS
from kaggriculture.policy.torch import strategy as S
from kaggriculture.policy.torch import tokenize
from kaggriculture.policy.torch.model import N_QUERY_SLOTS, N_UNIT_SLOTS, PolicyValueNet
from kaggriculture.rules import constants as C


def _candidate(op: int, arg: int, *, market: bool) -> V.SparseVector:
    vector = V.SparseVector()
    table = V.ACTION_MARKET_OP if market else V.ACTION_FARMER_OP
    vector.add(table.start + op)
    if arg >= 0:
        entity = arg
        if market and op == C.MARKET_OP_BUY_ANIMAL:
            entity = C.N_PRODUCTS + arg
        vector.add(V.ENTITY_ITEM.start + entity)
    return vector


def _unit_candidates() -> tuple[list[V.SparseVector], list[tuple[int, int]]]:
    item_ops = {C.FARMER_OP_PICKUP, C.FARMER_OP_PLANT, C.FARMER_OP_PLACE}
    metadata = [(op, -1) for op in range(C.N_FARMER_OPS) if op not in item_ops]
    metadata += [(C.FARMER_OP_PLANT, item) for item in range(C.N_CROPS)]
    metadata += [(C.FARMER_OP_PLACE, item) for item in range(C.N_SHED_ITEMS)]
    metadata += [(C.FARMER_OP_PICKUP, item) for item in range(C.N_SHED_ITEMS)]
    return [_candidate(op, arg, market=False) for op, arg in metadata], metadata


def _market_candidates() -> tuple[list[V.SparseVector], list[tuple[int, int]]]:
    metadata = [(C.MARKET_OP_HIRE, -1), (C.MARKET_OP_BUY_LAND, -1)]
    metadata += [(C.MARKET_OP_BUY_SEED, item) for item in range(C.N_CROPS)]
    metadata += [(C.MARKET_OP_BUY_ANIMAL, item) for item in range(C.N_ANIMALS)]
    metadata += [
        (C.MARKET_OP_BUY_PRODUCT, C.PRODUCTS.index(item)) for item in ("WHEAT", "FERTILIZER")
    ]
    metadata += [(C.MARKET_OP_SELL, item) for item in range(C.N_PRODUCTS)]
    vectors = [_candidate(op, arg, market=True) for op, arg in metadata]
    vectors += [A.market_wait_candidate(), A.market_stop_candidate()]
    # C.N_MARKET_OPS and +1 are the fixed-slot policy's WAIT and STOP pseudo-ops.
    metadata += [(C.N_MARKET_OPS, -1), (C.N_MARKET_OPS + 1, -1)]
    return vectors, metadata


UNIT_VECTORS, UNIT_META = _unit_candidates()
MARKET_VECTORS, MARKET_META = _market_candidates()
QUANTITY_VECTORS = A.quantity_candidates(V.MAX_ACTION_QUANTITY)


def _pack(vectors: list[V.SparseVector], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(vector.index) for vector in vectors), default=1)
    index = torch.zeros((len(vectors), width), dtype=torch.long, device=device)
    value = torch.zeros((len(vectors), width), dtype=torch.float32, device=device)
    for row, vector in enumerate(vectors):
        count = len(vector.index)
        if count:
            index[row, :count] = torch.tensor(vector.index, device=device)
            value[row, :count] = torch.tensor(vector.value, device=device)
    return index, value


def _key(vector: V.SparseVector) -> tuple[int, ...]:
    return tuple(vector.index)


def _item_name(op: int, arg: int, *, market: bool) -> str | None:
    if arg < 0:
        return None
    if market and op == C.MARKET_OP_BUY_ANIMAL:
        return C.ANIMALS[arg]
    if (market and op == C.MARKET_OP_BUY_SEED) or (not market and op == C.FARMER_OP_PLANT):
        return C.CROPS[arg]
    return C.SHED_ITEMS[arg]


def _copy_environment(obs: dict) -> tuple[dict, dict, dict, dict, list[dict]]:
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


@contextmanager
def _evaluation(net: PolicyValueNet):
    training = net.training
    net.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        net.train(training)


def _unit_quantity_upper_bound(
    op: int,
    arg: int,
    position: tuple[int, int],
    inventory: dict,
    farm: dict,
) -> int:
    """Return the Torch scoring bound corresponding to the JAX unit mask."""
    item = _item_name(op, arg, market=False)
    tile = farm["tiles"][position[1]][position[0]]
    if op == C.FARMER_OP_PLACE and not A.is_animal_placement(item, tile):
        return max(inventory.get(item, 0), 1)
    return len(QUANTITY_VECTORS)


def _execute_units(
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
    unit_ops = [UNIT_META[choice] for choice in selected_units[: len(positions)]]
    plant_counts = {crop: 0 for crop in C.CROPS}
    for op, arg in unit_ops:
        if op == C.FARMER_OP_PLANT:
            plant_counts[C.CROPS[arg]] += 1
    blocked_crops = {crop for crop, count in plant_counts.items() if count > seeds.get(crop, 0)}

    entries = []
    for slot, ((op, arg), position, inventory) in enumerate(
        zip(unit_ops, positions, inventories, strict=True)
    ):
        item = _item_name(op, arg, market=False)
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
        DS.commit_unit_action(
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


def predict_action(
    net: PolicyValueNet,
    obs: dict,
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1.0,
    counters: dict | None = None,
) -> dict:
    """Generate a legal action with one Transformer forward pass.

    Args:
        net: The policy network.
        obs: One Kaggriculture observation.
        turns_per_day: Number of turns in one game day.
        shed_capacity: Maximum total shed inventory.
        hire_mult: Multiplier applied to hand hiring costs.
        counters: Episode-history counters when the model uses history.

    Returns:
        A simulator-compatible action dictionary.

    Raises:
        ValueError: If critic or history settings are incompatible with submission inference.
    """
    if not net.actor_only:
        raise ValueError("submission inference requires an actor-only model")
    if counters is None:
        counters = {}
    device = next(net.parameters()).device
    encoder_index, encoder_value = _pack(
        tokenize.get_encoder_input(obs, turns_per_day, counters), device
    )
    player = obs["player"]
    farm_view = obs["farms"][player]
    positions_xy = [farm_view["farmer"], *farm_view["hands"]]
    active_count = len(positions_xy)
    positions = [y * V.BOARD_SIZE + x for x, y in positions_xy]
    positions += [L.NO_POSITION] * (N_UNIT_SLOTS - active_count)
    unit_positions = torch.tensor([positions], dtype=torch.long, device=device)
    unit_active = torch.zeros((1, N_UNIT_SLOTS), dtype=torch.bool, device=device)
    unit_active[:, :active_count] = True
    unit_inventory_index, unit_inventory_value = _pack(
        tokenize.get_unit_inventory_input(obs), device
    )
    farm, shed, seeds, market, inventories = _copy_environment(obs)

    with _evaluation(net):
        queries, _ = net.encode_actor(
            encoder_index[None],
            encoder_value[None],
            unit_positions,
            unit_active,
            unit_inventory_index[None],
            unit_inventory_value[None],
        )
        unit_hidden = queries[0, :N_UNIT_SLOTS]
        market_hidden = queries[0, N_UNIT_SLOTS:]

        legal_masks = []
        for slot, (position, inventory) in enumerate(zip(positions_xy, inventories, strict=True)):
            legal = {
                _key(vector)
                for vector in A.legal_unit_actions(
                    farm,
                    shed,
                    seeds,
                    inventory,
                    position,
                    obs["day"],
                    shed_capacity,
                    defer_shared_resources=True,
                )
            }
            strategic = S.unit_mask(obs, slot, UNIT_META)
            legal_masks.append(
                [
                    _key(vector) in legal and strategic[index]
                    for index, vector in enumerate(UNIT_VECTORS)
                ]
            )
        while len(legal_masks) < N_UNIT_SLOTS:
            legal_masks.append([index == 0 for index in range(len(UNIT_VECTORS))])
        unit_mask = torch.tensor(legal_masks, dtype=torch.bool, device=device)
        unit_logits = net.unit_logits(unit_hidden, unit_mask)
        selected_units = unit_logits.argmax(-1)

        market_mask = torch.tensor(S.market_mask(obs, MARKET_META), dtype=torch.bool, device=device)
        market_logits = net.market_logits(market_hidden, market_mask)
        selected_market = market_logits.argmax(-1)

        quantity_mask = torch.ones(
            (N_QUERY_SLOTS, len(QUANTITY_VECTORS)), dtype=torch.bool, device=device
        )
        numbers = torch.arange(1, len(QUANTITY_VECTORS) + 1, device=device)
        for slot, (position, inventory) in enumerate(zip(positions_xy, inventories, strict=True)):
            op, arg = UNIT_META[int(selected_units[slot])]
            maximum = _unit_quantity_upper_bound(op, arg, position, inventory, farm)
            quantity_mask[slot] = numbers <= maximum
        unit_quantity_logits = net.unit_quantity_logits(
            unit_hidden, selected_units, quantity_mask[:N_UNIT_SLOTS]
        )
        market_quantity_logits = net.market_quantity_logits(
            market_hidden, selected_market, quantity_mask[N_UNIT_SLOTS:]
        )
        quantity_logits = torch.cat([unit_quantity_logits, market_quantity_logits], dim=0)
        selected_quantities = quantity_logits.argmax(-1).add(1).tolist()
        selected_units = selected_units.tolist()
        selected_market = selected_market.tolist()

    unit_entries = _execute_units(
        selected_units,
        selected_quantities,
        positions_xy,
        inventories,
        farm,
        shed,
        seeds,
        obs["day"],
        turns_per_day,
        shed_capacity,
    )

    market_entries: list[list] = []
    active = True
    for offset, choice in enumerate(selected_market):
        if not active:
            break
        op, arg = MARKET_META[choice]
        if op == C.N_MARKET_OPS + 1:
            active = False
            continue
        if op == C.N_MARKET_OPS:
            market_entries.append(list(A.MARKET_WAIT_ACTION))
            continue
        item = _item_name(op, arg, market=True)
        legal = {
            _key(vector)
            for vector in A.legal_market_actions(farm, shed, market, hire_mult, shed_capacity)
        }
        if _key(MARKET_VECTORS[choice]) not in legal:
            market_entries.append(list(A.MARKET_WAIT_ACTION))
            continue
        op_name = C.MARKET_OP_NAMES[op]
        quantity = 1
        if A.requires_quantity(op_name, item):
            maximum = A.max_executable_quantity(op_name, item, farm, shed, market, shed_capacity)
            quantity = min(selected_quantities[N_UNIT_SLOTS + offset], maximum)
            if quantity <= 0:
                market_entries.append(list(A.MARKET_WAIT_ACTION))
                continue
        market_entries.append(_entry(op, item, quantity, market=True))
        DS.commit_market_action(
            farm, shed, market, op_name, item, quantity, shed_capacity, hire_mult
        )

    return {
        "farmer": unit_entries[0],
        "hands": unit_entries[1:],
        "market": market_entries,
    }
