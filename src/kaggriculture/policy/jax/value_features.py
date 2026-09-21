"""Critic専用の明示的なマクロ状態特徴。"""

from __future__ import annotations

import jax.numpy as jnp

from kaggriculture.rules import constants as C
from kaggriculture.simulator.assets import estimated_assets
from kaggriculture.simulator.state import State


def critic_macro_features(state: State, player: jnp.ndarray) -> jnp.ndarray:
    """1局面をplayer視点の正規化済みマクロ特徴へ変換する。

    Args:
        state: batch軸を持たない1局面の状態。
        player: 視点となるプレイヤー番号。

    Returns:
        自分・相手・差分の順に並べた現金、概算資産、土地数、hand数。
    """
    opponent = 1 - player
    money = state.money.astype(jnp.float32)
    assets = estimated_assets(state)
    land = jnp.sum(state.unlocked_quadrants[..., 1:], axis=-1).astype(jnp.float32)
    hands = jnp.sum(state.hands_active, axis=-1).astype(jnp.float32)

    def pair(values, scale):
        own = values[player] / scale
        other = values[opponent] / scale
        return own, other, own - other

    return jnp.asarray(
        [
            *pair(money, 200_000.0),
            *pair(assets, 300_000.0),
            *pair(land, float(C.N_QUADRANTS - 1)),
            *pair(hands, float(C.MAX_HANDS)),
        ],
        dtype=jnp.float32,
    )
