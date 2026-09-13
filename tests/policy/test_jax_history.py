"""JAX rollout用累積履歴のテスト。"""

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture.policy.jax import history as H
from kaggriculture.rules import constants as C
from kaggriculture.simulator.action import Action
from kaggriculture.simulator.reset import reset


def _pass_action(batch_size: int) -> Action:
    return Action(
        farmer_op=jnp.full((batch_size, 2), C.FARMER_OP_PASS),
        farmer_arg_idx=jnp.zeros((batch_size, 2), dtype=jnp.int32),
        farmer_n=jnp.ones((batch_size, 2), dtype=jnp.int32),
        hands_op=jnp.full((batch_size, 2, C.MAX_HANDS), C.FARMER_OP_PASS),
        hands_arg_idx=jnp.zeros((batch_size, 2, C.MAX_HANDS), dtype=jnp.int32),
        hands_n=jnp.ones((batch_size, 2, C.MAX_HANDS), dtype=jnp.int32),
        market_op=jnp.full((batch_size, 2, C.MAX_MARKET_ORDERS), -1),
        market_arg_idx=jnp.zeros((batch_size, 2, C.MAX_MARKET_ORDERS), dtype=jnp.int32),
        market_n=jnp.zeros((batch_size, 2, C.MAX_MARKET_ORDERS), dtype=jnp.int32),
    )


def test_update_counters_tracks_production_and_sale() -> None:
    state = reset(jax.random.key(0), 1)
    x, y = np.asarray(state.farmer_pos[0, 0])
    state = state._replace(
        tiles_kind=state.tiles_kind.at[0, 0, y, x].set(C.TILE_PLANT),
        tiles_crop_or_animal=state.tiles_crop_or_animal.at[0, 0, y, x].set(0),
        tiles_yield_units=state.tiles_yield_units.at[0, 0, y, x].set(3),
        shed=state.shed.at[0, 0, 0].set(5),
    )
    action = _pass_action(1)
    action = action._replace(
        farmer_op=action.farmer_op.at[0, 0].set(C.FARMER_OP_HARVEST),
        market_op=action.market_op.at[0, 0, 0].set(C.MARKET_OP_SELL),
        market_n=action.market_n.at[0, 0, 0].set(3),
    )
    counters = jax.jit(H.update_counters)(state, action, H.zeros(1))

    assert counters.produced[0, 0, 0] == 3
    assert counters.sold[0, 0, 0] == 3
    assert counters.estimated_revenue[0, 0, 0] > 0
    assert counters.has_ever_sold[0, 0, 0]
