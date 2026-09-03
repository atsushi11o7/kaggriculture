"""植物の寿命超過後の減衰(元コードの _decay_plants)。

タイルごとに独立した要素ごとの処理なので、盤面全体の配列にそのまま適用できる
(vmap不要)。
"""

import jax.numpy as jnp

from . import constants as C
from .constants import is_plant as _is_plant


def apply_decay(tile_kind, tile_crop_or_animal, tile_max_lifespan_step, tile_yield_units, step):
    """植物の寿命超過後の減衰を盤面全体に適用する。

    PLANTタイルで、max_lifespan_step >= 0 かつ step >= max_lifespan_step かつ
    (step - max_lifespan_step)が偶数の時、yield_unitsを1減らす。0以下になったら
    WEEDになる。

    Args:
        tile_kind: constants.TILE_*。
        tile_crop_or_animal: 現在の作物インデックス。
        tile_max_lifespan_step: 寿命ステップ(-1 = 該当なし)。
        tile_yield_units: 現在の収穫可能量。
        step: 現在の通しターン数。

    Returns:
        (new_tile_kind, new_tile_crop_or_animal, new_tile_yield_units)。
    """
    is_plant = _is_plant(tile_kind)
    has_lifespan = tile_max_lifespan_step >= 0
    past_lifespan = step >= tile_max_lifespan_step
    is_decay_tick = ((step - tile_max_lifespan_step) % 2) == 0
    applies = is_plant & has_lifespan & past_lifespan & is_decay_tick

    new_yield_units = jnp.where(applies, tile_yield_units - 1, tile_yield_units)
    becomes_weed = applies & (new_yield_units <= 0)

    new_tile_kind = jnp.where(becomes_weed, C.TILE_WEED, tile_kind)
    new_tile_crop_or_animal = jnp.where(becomes_weed, -1, tile_crop_or_animal)

    return new_tile_kind, new_tile_crop_or_animal, new_yield_units
