"""農夫/hand 1体分の行動の統合ディスパッチ(元コードの _apply_unit_action)。

個々のopの実装はcrop_actions.py(PLANT/WATER/HARVEST/FERTILIZE/DIG)、
animal_actions.py(BUILD_*/PLACE/FEED/CARE/COLLECT_FERTILIZER)、
inventory_actions.py(PICKUP/DROP)に分かれている。このモジュールはそれらを
1ユニット分の行動として合成する層のみを持つ。

各関数は1ユニット・バッチ軸なしの純粋関数。バッチ適用時は呼び出し側でvmapする。
"""

import jax.numpy as jnp

from . import constants as C
from .animal_actions import (
    apply_build,
    apply_care,
    apply_collect_fertilizer,
    apply_feed,
    apply_place,
)
from .crop_actions import apply_dig, apply_fertilize, apply_harvest, apply_plant, apply_water
from .inventory_actions import apply_drop, apply_pickup

_MOVE_DELTA = jnp.array(C.FARMER_OP_MOVE_DELTA, dtype=jnp.int32)


def apply_movement(pos, op, board_size):
    """1ユニット分の移動を適用する。

    移動系(NORTH/SOUTH/EAST/WEST)以外は不動。盤面外に出る移動はno-op。
    LOCKEDタイルへの移動は許可する(踏めるだけなので判定しない)。

    Args:
        pos: [2] (x, y) の現在位置。
        op: constants.FARMER_OP_*。
        board_size: 盤面の一辺のサイズ。

    Returns:
        [2] 新しい位置。
    """
    new_pos = pos + _MOVE_DELTA[op]
    in_bounds = jnp.all((new_pos >= 0) & (new_pos < board_size))
    return jnp.where(in_bounds, new_pos, pos)


def apply_unit_action(
    op,
    arg_idx,
    n,
    pos,
    board_size,
    tile_kind,
    tile_crop_or_animal,
    tile_planted_or_placed_day,
    tile_watered_or_fed_today,
    tile_consecutive_unwatered_or_unfed,
    tile_yield_units,
    tile_max_lifespan_step,
    tile_fertilized_until_day,
    tile_cared_today,
    tile_fertilizer_available,
    tile_pending_care_bonus,
    day,
    turns_per_day,
    shed,
    inventory,
    shed_capacity,
    blocked,
):
    """1ユニット分の行動を全て適用する(元コードの_apply_unit_action)。

    各opの処理は個々の関数(apply_water, apply_harvest, ...)に委譲する。opは
    1ユニットにつき1つしか無いため、該当しない関数は入力をそのまま返す
    (no-op)。よって呼び出し順序に関わらず正しく合成できる。

    arg_idxの意味はopによって異なる: PLANTなら作物インデックス(constants.CROPS)、
    PICKUP/PLACEなら品目インデックス(constants.SHED_ITEMS)。それ以外のopでは
    無視される。

    Args:
        op: constants.FARMER_OP_*。
        arg_idx: op依存の引数(上記参照)。
        n: PICKUP/PLACEの要求数。
        pos: [2] (x, y) 現在位置。
        board_size: 盤面の一辺のサイズ。
        tile_kind: constants.TILE_*。
        tile_crop_or_animal: 現在の作物/動物インデックス。
        tile_planted_or_placed_day: 現在の植付/配置日。
        tile_watered_or_fed_today: 現在の水やり/給餌済みフラグ。
        tile_consecutive_unwatered_or_unfed: 現在の未水やり/未給餌連続日数。
        tile_yield_units: 現在の収穫可能量。
        tile_max_lifespan_step: 現在の寿命ステップ。
        tile_fertilized_until_day: 現在の施肥ボーナス最終日。
        tile_cared_today: 現在の世話済みフラグ。
        tile_fertilizer_available: 現在の未回収肥料フラグ。
        tile_pending_care_bonus: 現在の世話ボーナス蓄積値。
        day: 現在の日。
        turns_per_day: 1日のターン数。
        shed: [N_SHED_ITEMS] 納屋の在庫。
        inventory: [N_SHED_ITEMS] このユニットの持ち物。
        shed_capacity: 納屋の容量。
        blocked: [N_CROPS] bool。compute_plant_blockの結果。

        Returns:
        (new_pos, new_tile_kind, new_tile_crop_or_animal,
        new_tile_planted_or_placed_day, new_tile_watered_or_fed_today,
        new_tile_consecutive_unwatered_or_unfed, new_tile_yield_units,
        new_tile_max_lifespan_step, new_tile_fertilized_until_day,
        new_tile_cared_today, new_tile_fertilizer_available,
        new_tile_pending_care_bonus, new_shed, new_inventory, seed_used)。
    """
    new_pos = apply_movement(pos, op, board_size)

    tile_watered_or_fed_today, tile_yield_units = apply_water(
        op,
        tile_kind,
        tile_crop_or_animal,
        tile_watered_or_fed_today,
        tile_planted_or_placed_day,
        tile_fertilized_until_day,
        tile_yield_units,
        day,
    )

    tile_kind, tile_crop_or_animal, tile_yield_units, inventory = apply_harvest(
        op,
        tile_kind,
        tile_crop_or_animal,
        tile_planted_or_placed_day,
        tile_yield_units,
        day,
        inventory,
    )

    (
        tile_kind,
        tile_crop_or_animal,
        tile_planted_or_placed_day,
        tile_watered_or_fed_today,
        tile_consecutive_unwatered_or_unfed,
        tile_yield_units,
        tile_max_lifespan_step,
        tile_fertilized_until_day,
        seed_used,
    ) = apply_plant(
        op,
        arg_idx,
        blocked,
        tile_kind,
        tile_crop_or_animal,
        tile_planted_or_placed_day,
        tile_watered_or_fed_today,
        tile_consecutive_unwatered_or_unfed,
        tile_yield_units,
        tile_max_lifespan_step,
        tile_fertilized_until_day,
        day,
        turns_per_day,
    )

    tile_fertilized_until_day, inventory = apply_fertilize(
        op, tile_kind, tile_fertilized_until_day, day, inventory
    )

    tile_kind, tile_crop_or_animal = apply_dig(op, tile_kind, tile_crop_or_animal)

    tile_kind = apply_build(op, tile_kind)

    tile_watered_or_fed_today, inventory = apply_feed(
        op, tile_kind, tile_crop_or_animal, tile_watered_or_fed_today, inventory
    )

    tile_fertilizer_available, inventory = apply_collect_fertilizer(
        op, tile_kind, tile_crop_or_animal, tile_fertilizer_available, inventory
    )

    tile_cared_today = apply_care(op, tile_kind, tile_crop_or_animal, tile_cared_today)

    shed, inventory = apply_pickup(op, arg_idx, n, pos, board_size, shed, inventory)

    shed, inventory = apply_drop(op, pos, board_size, shed, inventory, shed_capacity)

    (
        tile_crop_or_animal,
        tile_planted_or_placed_day,
        tile_yield_units,
        tile_watered_or_fed_today,
        tile_consecutive_unwatered_or_unfed,
        tile_cared_today,
        tile_fertilizer_available,
        tile_pending_care_bonus,
        shed,
        inventory,
    ) = apply_place(
        op,
        arg_idx,
        n,
        pos,
        board_size,
        tile_kind,
        tile_crop_or_animal,
        tile_planted_or_placed_day,
        tile_yield_units,
        tile_watered_or_fed_today,
        tile_consecutive_unwatered_or_unfed,
        tile_cared_today,
        tile_fertilizer_available,
        tile_pending_care_bonus,
        day,
        shed,
        inventory,
        shed_capacity,
    )

    return (
        new_pos,
        tile_kind,
        tile_crop_or_animal,
        tile_planted_or_placed_day,
        tile_watered_or_fed_today,
        tile_consecutive_unwatered_or_unfed,
        tile_yield_units,
        tile_max_lifespan_step,
        tile_fertilized_until_day,
        tile_cared_today,
        tile_fertilizer_available,
        tile_pending_care_bonus,
        shed,
        inventory,
        seed_used,
    )
