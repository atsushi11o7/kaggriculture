"""JAX endgame action overrides for GPU rollout."""

from __future__ import annotations

import jax.numpy as jnp

from kaggriculture.rules import constants as C
from kaggriculture.simulator.action import Action
from kaggriculture.simulator.state import State


def _return_ops(positions: jnp.ndarray, board_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    left = board_size // 2 - 1
    right = board_size // 2
    targets = jnp.asarray(((left, left), (right, left), (left, right), (right, right)))
    distances = jnp.abs(positions[:, None, :] - targets[None, :, :]).sum(-1)
    target = targets[jnp.argmin(distances, axis=1)]
    dx = target[:, 0] - positions[:, 0]
    dy = target[:, 1] - positions[:, 1]
    op = jnp.where(
        dx > 0,
        C.FARMER_OP_EAST,
        jnp.where(
            dx < 0,
            C.FARMER_OP_WEST,
            jnp.where(dy > 0, C.FARMER_OP_SOUTH, C.FARMER_OP_NORTH),
        ),
    )
    op = jnp.where(distances.min(-1) == 0, C.FARMER_OP_DROP, op)
    return op, distances.min(-1)


def apply(
    state: State,
    player: jnp.ndarray,
    action: Action,
    *,
    episode_steps: int = 720,
    board_size: int = 10,
) -> tuple[Action, jnp.ndarray, jnp.ndarray]:
    """Apply endgame overrides and return overridden unit/market slot masks."""
    positions = jnp.concatenate([state.farmer_pos[player][None], state.hands_pos[player]], axis=0)
    inventories = jnp.concatenate(
        [state.farmer_inventory[player][None], state.hands_inventory[player]], axis=0
    )
    active = jnp.concatenate([jnp.ones((1,), dtype=bool), state.hands_active[player]], axis=0)
    return_op, distance = _return_ops(positions, board_size)
    cargo = jnp.any(inventories[:, : C.N_PRODUCTS] > 0, axis=-1)
    remaining = episode_steps - 2 - state.step + 1
    override_unit = active & cargo & (distance + 1 >= remaining) & (remaining > 0)

    unit_op = jnp.concatenate([action.farmer_op[None], action.hands_op])
    unit_arg = jnp.concatenate([action.farmer_arg_idx[None], action.hands_arg_idx])
    unit_n = jnp.concatenate([action.farmer_n[None], action.hands_n])
    unit_op = jnp.where(override_unit, return_op, unit_op)
    unit_arg = jnp.where(override_unit, 0, unit_arg)
    unit_n = jnp.where(override_unit, 1, unit_n)

    final = state.step == episode_steps - 2
    at_shed = active & (distance == 0)
    delivered = jnp.sum(inventories[:, : C.N_PRODUCTS] * at_shed[:, None], axis=0)
    available = state.shed[player, : C.N_PRODUCTS] + delivered
    products = jnp.nonzero(available > 0, size=C.MAX_MARKET_ORDERS, fill_value=0)[0]
    count = jnp.sum(available > 0)
    liquidation = jnp.arange(C.MAX_MARKET_ORDERS) < count
    liquidate_op = jnp.where(liquidation, C.MARKET_OP_SELL, -1)
    liquidate_arg = jnp.where(liquidation, products, 0)
    liquidate_n = jnp.where(liquidation, jnp.minimum(available[products], 100), 0)
    market_op = jnp.where(final, liquidate_op, action.market_op)
    market_arg = jnp.where(final, liquidate_arg, action.market_arg_idx)
    market_n = jnp.where(final, liquidate_n, action.market_n)
    result = Action(
        unit_op[0],
        unit_arg[0],
        unit_n[0],
        unit_op[1:],
        unit_arg[1:],
        unit_n[1:],
        market_op,
        market_arg,
        market_n,
    )
    return result, override_unit, jnp.full((C.MAX_MARKET_ORDERS,), final)
