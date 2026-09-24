"""JAX方策へ適用する戦略上の候補制約。"""

import jax.numpy as jnp

from kaggriculture.policy.common.strategy import market_candidate_allowed
from kaggriculture.policy.jax import actions as A
from kaggriculture.rules import constants as C
from kaggriculture.rules import game_params as P
from kaggriculture.simulator.state import State

_CROP_FIRST_YIELD_DAY = jnp.asarray(P.CROP_FIRST_YIELD_DAY)
_ANIMAL_FIRST_YIELD_DAY = jnp.asarray(P.ANIMAL_FIRST_YIELD_DAY)


def unit_mask(state: State, player: jnp.ndarray, unit: jnp.ndarray) -> jnp.ndarray:
    """unit候補の戦略maskを返す。

    Args:
        state: ゲーム状態。
        player: 対象プレイヤー。
        unit: 対象unit slot。

    Returns:
        候補順の許可mask。現時点では全候補を許可する。
    """
    del player, unit
    day = state.step // 24
    final_day = (720 - 2) // 24
    op = A.UNIT_CANDIDATES.op
    arg = jnp.clip(A.UNIT_CANDIDATES.arg, 0, C.N_CROPS - 1)
    plant_in_time = day + _CROP_FIRST_YIELD_DAY[arg] < final_day
    return (op != C.FARMER_OP_PLANT) | plant_in_time


def market_mask(state: State, player: jnp.ndarray) -> jnp.ndarray:
    """市場slotと候補の戦略maskを返す。

    Args:
        state: ゲーム状態。
        player: 対象プレイヤー。

    Returns:
        市場slotと候補の許可mask。
    """
    del player
    allowed = [
        [
            market_candidate_allowed(
                slot, int(op), max_slots=C.MAX_MARKET_ORDERS, wait_op=A.MARKET_WAIT
            )
            for op in A.MARKET_CANDIDATES.op.tolist()
        ]
        for slot in range(C.MAX_MARKET_ORDERS)
    ]
    allowed = jnp.asarray(allowed, dtype=bool)
    day = state.step // 24
    final_day = (720 - 2) // 24
    op = A.MARKET_CANDIDATES.op
    arg = A.MARKET_CANDIDATES.arg
    crop = jnp.clip(arg, 0, C.N_CROPS - 1)
    animal = jnp.clip(arg, 0, C.N_ANIMALS - 1)
    in_time = jnp.ones_like(op, dtype=bool)
    in_time &= (op != C.MARKET_OP_BUY_SEED) | (day + _CROP_FIRST_YIELD_DAY[crop] < final_day)
    in_time &= (op != C.MARKET_OP_BUY_ANIMAL) | (day + _ANIMAL_FIRST_YIELD_DAY[animal] < final_day)
    in_time &= (op != C.MARKET_OP_BUY_LAND) | (day + jnp.min(_CROP_FIRST_YIELD_DAY) < final_day)
    return allowed & in_time[None]
