"""CPU inference for the fixed-slot policy without a JAX dependency."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.endgame import torch as G
from kaggriculture.policy.torch import actions as A
from kaggriculture.policy.torch import candidates as K
from kaggriculture.policy.torch import executor as E
from kaggriculture.policy.torch import strategy as S
from kaggriculture.policy.torch import tokenize
from kaggriculture.policy.torch.model import N_QUERY_SLOTS, N_UNIT_SLOTS, PolicyValueNet


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


@contextmanager
def _evaluation(net: PolicyValueNet):
    training = net.training
    net.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        net.train(training)


def _positions(obs: dict, device: torch.device):
    """Return active positions and padded position tensors."""
    player = obs["player"]
    farm = obs["farms"][player]
    positions_xy = [farm["farmer"], *farm["hands"]]
    positions = [y * V.BOARD_SIZE + x for x, y in positions_xy]
    positions += [L.NO_POSITION] * (N_UNIT_SLOTS - len(positions_xy))
    unit_positions = torch.tensor([positions], dtype=torch.long, device=device)
    unit_active = torch.zeros((1, N_UNIT_SLOTS), dtype=torch.bool, device=device)
    unit_active[:, : len(positions_xy)] = True
    return positions_xy, unit_positions, unit_active


def _unit_masks(
    obs: dict,
    positions: list[tuple[int, int]],
    inventories: list[dict],
    shed: dict,
    seeds: dict,
    *,
    shed_capacity: int,
) -> list[list[bool]]:
    """Build legal and strategic masks for all padded unit slots."""
    farm = obs["farms"][obs["player"]]
    masks = []
    for slot, (position, inventory) in enumerate(zip(positions, inventories, strict=True)):
        legal = {
            K.key(vector)
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
        strategic = S.unit_mask(obs, slot, K.UNIT_META)
        masks.append(
            [
                K.key(vector) in legal and strategic[index]
                for index, vector in enumerate(K.UNIT_VECTORS)
            ]
        )
    padding = [index == 0 for index in range(len(K.UNIT_VECTORS))]
    masks.extend([padding] * (N_UNIT_SLOTS - len(masks)))
    return masks


def predict_action(
    net: PolicyValueNet,
    obs: dict,
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1.0,
    episode_steps: int = 720,
    board_size: int = 10,
    counters: dict | None = None,
) -> dict:
    """Generate one simulator-compatible action with one Transformer forward pass.

    Args:
        net: Actor-only policy network.
        obs: One Kaggriculture observation.
        turns_per_day: Number of turns in one game day.
        shed_capacity: Maximum total shed inventory.
        hire_mult: Multiplier applied to hand hiring costs.
        episode_steps: Number of steps in one game.
        board_size: Width and height of the square board.
        counters: Episode-history counters.

    Returns:
        A simulator-compatible action dictionary.

    Raises:
        ValueError: If the network contains training-only critic parameters.
    """
    if not net.actor_only:
        raise ValueError("submission inference requires an actor-only model")
    counters = counters or {}
    device = next(net.parameters()).device
    encoder_index, encoder_value = _pack(
        tokenize.get_encoder_input(obs, turns_per_day, counters), device
    )
    positions_xy, unit_positions, unit_active = _positions(obs, device)
    inventory_index, inventory_value = _pack(tokenize.get_unit_inventory_input(obs), device)
    _, shed, seeds, _, inventories = E.copy_environment(obs)

    with _evaluation(net):
        queries, _ = net.encode_actor(
            encoder_index[None],
            encoder_value[None],
            unit_positions,
            unit_active,
            inventory_index[None],
            inventory_value[None],
        )
        unit_hidden = queries[0, :N_UNIT_SLOTS]
        market_hidden = queries[0, N_UNIT_SLOTS:]
        unit_mask = torch.tensor(
            _unit_masks(
                obs,
                positions_xy,
                inventories,
                shed,
                seeds,
                shed_capacity=shed_capacity,
            ),
            dtype=torch.bool,
            device=device,
        )
        market_mask = torch.tensor(
            S.market_mask(obs, K.MARKET_META), dtype=torch.bool, device=device
        )
        selected_units = net.unit_logits(unit_hidden, unit_mask).argmax(-1)
        selected_market = net.market_logits(market_hidden, market_mask).argmax(-1)

        quantity_mask = torch.ones(
            (N_QUERY_SLOTS, len(K.QUANTITY_VECTORS)), dtype=torch.bool, device=device
        )
        numbers = torch.arange(1, len(K.QUANTITY_VECTORS) + 1, device=device)
        farm = obs["farms"][obs["player"]]
        for slot, (position, inventory) in enumerate(zip(positions_xy, inventories, strict=True)):
            op, arg = K.UNIT_META[int(selected_units[slot])]
            maximum = E.unit_quantity_upper_bound(op, arg, position, inventory, farm)
            quantity_mask[slot] = numbers <= maximum
        unit_quantities = net.unit_quantity_logits(
            unit_hidden, selected_units, quantity_mask[:N_UNIT_SLOTS]
        )
        market_quantities = net.market_quantity_logits(
            market_hidden, selected_market, quantity_mask[N_UNIT_SLOTS:]
        )
        selected_quantities = torch.cat([unit_quantities, market_quantities]).argmax(-1).add(1)

    action = E.resolve_action(
        obs,
        selected_units.tolist(),
        selected_market.tolist(),
        selected_quantities.tolist(),
        turns_per_day=turns_per_day,
        shed_capacity=shed_capacity,
        hire_mult=hire_mult,
    )
    return G.apply(
        obs,
        action,
        episode_steps=episode_steps,
        turns_per_day=turns_per_day,
        board_size=board_size,
    )
