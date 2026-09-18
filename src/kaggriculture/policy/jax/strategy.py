"""JAX方策へ適用する戦略上の候補制約。"""

import jax.numpy as jnp

from kaggriculture.policy.common.strategy import market_candidate_allowed
from kaggriculture.policy.jax import actions as A
from kaggriculture.rules import constants as C
from kaggriculture.simulator.state import State


def unit_mask(state: State, player: jnp.ndarray, unit: jnp.ndarray) -> jnp.ndarray:
    """unit候補の戦略maskを返す。

    Args:
        state: ゲーム状態。
        player: 対象プレイヤー。
        unit: 対象unit slot。

    Returns:
        候補順の許可mask。現時点では全候補を許可する。
    """
    del state, player, unit
    return jnp.ones_like(A.UNIT_CANDIDATES.op, dtype=bool)


def market_mask(state: State, player: jnp.ndarray) -> jnp.ndarray:
    """市場slotと候補の戦略maskを返す。

    Args:
        state: ゲーム状態。
        player: 対象プレイヤー。

    Returns:
        市場slotと候補の許可mask。
    """
    del state, player
    allowed = [
        [
            market_candidate_allowed(
                slot, int(op), max_slots=C.MAX_MARKET_ORDERS, wait_op=A.MARKET_WAIT
            )
            for op in A.MARKET_CANDIDATES.op.tolist()
        ]
        for slot in range(C.MAX_MARKET_ORDERS)
    ]
    return jnp.asarray(allowed, dtype=bool)
