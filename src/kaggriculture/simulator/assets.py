"""報酬とcriticで共有する状態の概算資産評価。"""

from __future__ import annotations

import jax.numpy as jnp

from kaggriculture.rules import constants as C
from kaggriculture.rules import game_params as G
from kaggriculture.simulator.state import State

_PRODUCT_PRICES = jnp.asarray(G.MARKET_BASE_PRICE, dtype=jnp.float32)
_SEED_COSTS = jnp.asarray(G.CROP_SEED_COST, dtype=jnp.float32)
_ANIMAL_COSTS = jnp.asarray(G.ANIMAL_COST, dtype=jnp.float32)
_LAND_COSTS = jnp.asarray(G.LAND_PRICES, dtype=jnp.float32)


def estimated_assets(state: State) -> jnp.ndarray:
    """両プレイヤーの概算資産を現金単位で返す。

    Args:
        state: 単一またはbatch化されたシミュレータ状態。

    Returns:
        最終軸が両プレイヤーの概算資産。handは日末に残らないため含めず、
        在庫は市場操作による自己増幅を避けるため固定基準価格で評価する。
    """
    holdings = state.shed + state.farmer_inventory + jnp.sum(state.hands_inventory, axis=-2)
    product_value = jnp.sum(holdings[..., : C.N_PRODUCTS] * _PRODUCT_PRICES, axis=-1)
    animal_value = jnp.sum(holdings[..., C.N_PRODUCTS :] * _ANIMAL_COSTS, axis=-1)
    seed_value = jnp.sum(state.seeds * _SEED_COSTS, axis=-1)
    land_value = jnp.sum(state.unlocked_quadrants[..., 1:] * _LAND_COSTS, axis=-1)

    item = state.tiles_crop_or_animal
    crop_value = _SEED_COSTS[jnp.clip(item, 0, C.N_CROPS - 1)]
    animal_tile_value = _ANIMAL_COSTS[jnp.clip(item, 0, C.N_ANIMALS - 1)]
    planted = state.tiles_kind == C.TILE_PLANT
    occupied = ((state.tiles_kind == C.TILE_COOP) | (state.tiles_kind == C.TILE_PASTURE)) & (
        item >= 0
    )
    tile_value = jnp.sum(
        jnp.where(planted, crop_value, 0.0) + jnp.where(occupied, animal_tile_value, 0.0),
        axis=(-2, -1),
    )
    return state.money + product_value + animal_value + seed_value + land_value + tile_value
