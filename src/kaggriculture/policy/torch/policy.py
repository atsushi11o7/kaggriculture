"""CPU inference for the fixed-slot policy without a JAX dependency."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.torch import actions as A
from kaggriculture.policy.torch import tokenize
from kaggriculture.policy.torch.model import (
    N_MARKET_SLOTS,
    N_UNIT_SLOTS,
    PolicyValueNet,
)
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


def predict_action(
    net: PolicyValueNet,
    obs: dict,
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1.0,
    counters: dict | None = None,
) -> dict:
    """Generate a legal action with one Transformer forward pass."""
    if net.uses_asymmetric_critic:
        raise ValueError("submission actor must have use_asymmetric_critic=False")
    if net.uses_episode_history and counters is None:
        raise ValueError("episode history model requires counters")
    if not net.uses_episode_history and counters is not None:
        raise ValueError("model does not use episode history")
    device = next(net.parameters()).device
    encoder_index, encoder_value = _pack(
        tokenize.get_encoder_input(obs, turns_per_day, counters), device
    )
    encoder_index = encoder_index[None]
    encoder_value = encoder_value[None]
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
    unit_inventory_index = unit_inventory_index[None]
    unit_inventory_value = unit_inventory_value[None]

    with _evaluation(net):
        queries, _ = net(
            encoder_index,
            encoder_value,
            unit_positions,
            unit_active,
            unit_inventory_index,
            unit_inventory_value,
        )
        unit_index, unit_value = _pack(UNIT_VECTORS, device)
        market_index, market_value = _pack(MARKET_VECTORS, device)
        unit_hidden = queries[0, :N_UNIT_SLOTS]
        market_hidden = queries[0, N_UNIT_SLOTS:]

        farm, shed, seeds, market, inventories = _copy_environment(obs)
        legal_masks = []
        for position, inventory in zip(positions_xy, inventories, strict=True):
            legal = {
                _key(vector)
                for vector in A.legal_unit_actions(
                    farm, shed, seeds, inventory, position, obs["day"], shed_capacity
                )
            }
            legal_masks.append([_key(vector) in legal for vector in UNIT_VECTORS])
        while len(legal_masks) < N_UNIT_SLOTS:
            legal_masks.append([index == 0 for index in range(len(UNIT_VECTORS))])
        unit_mask = torch.tensor(legal_masks, dtype=torch.bool, device=device)
        unit_logits = net.score_candidates(unit_hidden, unit_index, unit_value, unit_mask)
        selected_units = unit_logits.argmax(-1).tolist()

        market_mask = torch.ones(
            (N_MARKET_SLOTS, len(MARKET_VECTORS)), dtype=torch.bool, device=device
        )
        market_logits = net.score_candidates(market_hidden, market_index, market_value, market_mask)
        selected_market = market_logits.argmax(-1).tolist()

        quantity_index, quantity_value = _pack(QUANTITY_VECTORS, device)
        selected_vectors = torch.cat(
            [unit_index[selected_units], market_index[selected_market]], dim=0
        )
        selected_values = torch.cat(
            [unit_value[selected_units], market_value[selected_market]], dim=0
        )
        # Both fixed candidate tables have width two.
        conditioned = net.condition_quantity(queries[0], selected_vectors, selected_values)
        quantity_mask = torch.ones(
            (conditioned.shape[0], V.MAX_ACTION_QUANTITY), dtype=torch.bool, device=device
        )
        quantity_logits = net.score_candidates(
            conditioned, quantity_index, quantity_value, quantity_mask
        )
        selected_quantities = quantity_logits.argmax(-1).add(1).tolist()

    unit_entries: list[list] = []
    plant_counts = {crop: 0 for crop in C.CROPS}
    for slot, choice in enumerate(selected_units[:active_count]):
        op, arg = UNIT_META[choice]
        item = _item_name(op, arg, market=False)
        tile = farm["tiles"][positions_xy[slot][1]][positions_xy[slot][0]]
        needs = A.requires_quantity(C.FARMER_OP_NAMES[op], item, tile)
        quantity = 1
        if needs:
            maximum = A.max_executable_quantity(
                C.FARMER_OP_NAMES[op],
                item,
                farm,
                shed,
                None,
                shed_capacity,
                inventory=inventories[slot],
            )
            quantity = min(selected_quantities[slot], maximum)
        if op == C.FARMER_OP_PLANT and item is not None:
            plant_counts[item] += 1
        unit_entries.append(_entry(op, item, max(quantity, 1), market=False))
    for crop, count in plant_counts.items():
        if count > seeds.get(crop, 0):
            for slot, entry in enumerate(unit_entries):
                if entry[:2] == ["PLANT", crop]:
                    unit_entries[slot] = ["PASS"]

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
        needs = A.requires_quantity(op_name, item)
        quantity = 1
        if needs:
            maximum = A.max_executable_quantity(op_name, item, farm, shed, market, shed_capacity)
            quantity = min(selected_quantities[N_UNIT_SLOTS + offset], maximum)
            if quantity <= 0:
                market_entries.append(list(A.MARKET_WAIT_ACTION))
                continue
        market_entries.append(_entry(op, item, quantity, market=True))
        from kaggriculture.policy.torch.decode import commit_market_action

        commit_market_action(farm, shed, market, op_name, item, quantity, shed_capacity, hire_mult)

    return {
        "farmer": unit_entries[0],
        "hands": unit_entries[1:],
        "market": market_entries,
    }
