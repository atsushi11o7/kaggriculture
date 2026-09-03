"""作物関連の行動(元コードの_apply_unit_action内、PLANT/WATER/HARVEST/FERTILIZE/DIG)。

各関数は1ユニット・バッチ軸なしの純粋関数。バッチ適用時は呼び出し側でvmapする。
"""

import jax
import jax.numpy as jnp

from . import constants as C
from . import game_params as P
from .constants import is_animal_structure as _is_animal_structure
from .constants import is_plant as _is_plant
from .game_params import CROP_FIRST_YIELD_DAY_JAX as _CROP_FIRST_YIELD_DAY
from .game_params import CROP_IS_ONGOING_JAX as _CROP_IS_ONGOING
from .game_params import CROP_MAX_YIELD_JAX as _CROP_MAX_YIELD

_CROP_MAX_YIELD_DAY = jnp.array(P.CROP_MAX_YIELD_DAY, dtype=jnp.int32)
_CROP_PRODUCT_IDX = jnp.array(P.CROP_PRODUCT_IDX, dtype=jnp.int32)
_ANIMAL_PRODUCT_IDX = jnp.array(P.ANIMAL_PRODUCT_IDX, dtype=jnp.int32)
_FERTILIZER_IDX = C.SHED_ITEMS.index("FERTILIZER")


def apply_water(
    op,
    tile_kind,
    tile_crop,
    tile_watered_today,
    tile_planted_day,
    tile_fertilized_until_day,
    tile_yield_units,
    day,
):
    """WATER行動を1タイルに適用する。

    対象がPLANTでない、または既に水やり済みなら何もしない。一発収穫型のみ、
    ボーナス窓(ceil(max_yield_day/2)日目〜max_yield_day日目)の水やりで
    yield_unitsが増える(施肥中は+2、そうでなければ+1、max_yieldで頭打ち)。
    継続収穫型はここでは増えない(日次更新側で処理する)。

    Args:
        op: constants.FARMER_OP_*。
        tile_kind: constants.TILE_*。
        tile_crop: 作物インデックス(constants.CROPS)。
        tile_watered_today: 既に水やり済みか。
        tile_planted_day: 植えた日。
        tile_fertilized_until_day: 施肥ボーナスが有効な最終日。
        tile_yield_units: 現在の収穫可能量。
        day: 現在の日。

    Returns:
        (new_watered_today, new_yield_units)。
    """
    is_water_op = op == C.FARMER_OP_WATER
    is_plant = _is_plant(tile_kind)
    not_watered_yet = jnp.logical_not(tile_watered_today)
    applies = is_water_op & is_plant & not_watered_yet

    new_watered_today = jnp.where(applies, True, tile_watered_today)

    is_ongoing = _CROP_IS_ONGOING[tile_crop]
    max_yield_day = _CROP_MAX_YIELD_DAY[tile_crop]
    max_yield = _CROP_MAX_YIELD[tile_crop]

    age_days = day - tile_planted_day
    window_start = (max_yield_day + 1) // 2  # ceil(max_yield_day / 2)
    in_bonus_window = (age_days >= window_start) & (age_days <= max_yield_day)
    is_fertilized = tile_fertilized_until_day >= day
    bonus = jnp.where(is_fertilized, 2, 1)

    gives_bonus = applies & jnp.logical_not(is_ongoing) & in_bonus_window
    new_yield_units = jnp.where(
        gives_bonus,
        jnp.minimum(max_yield, tile_yield_units + bonus),
        tile_yield_units,
    )

    return new_watered_today, new_yield_units


def apply_harvest(
    op,
    tile_kind,
    tile_crop_or_animal,
    tile_planted_or_placed_day,
    tile_yield_units,
    day,
    inventory,
):
    """HARVEST行動を1タイルに適用する。

    植物はplanted_dayからfirst_yield_day経過が必要。一発収穫型は収穫後に
    タイルが更地(TILE_EMPTY)に戻り、継続収穫型はタイルを維持する。動物は
    日数チェック無し(yield_unitsはfirst_yield_day経過後にしか積まれないため)。
    yield_units<=0なら何もしない。

    Args:
        op: constants.FARMER_OP_*。
        tile_kind: constants.TILE_*。
        tile_crop_or_animal: 作物または動物インデックス(-1 = 該当なし)。
        tile_planted_or_placed_day: 植えた/配置した日。
        tile_yield_units: 現在の収穫可能量。
        day: 現在の日。
        inventory: [N_SHED_ITEMS] このユニットの持ち物。

    Returns:
        (new_tile_kind, new_tile_crop_or_animal, new_yield_units, new_inventory)。
    """
    is_harvest_op = op == C.FARMER_OP_HARVEST
    has_yield = tile_yield_units > 0
    is_plant = _is_plant(tile_kind)
    is_animal_structure = _is_animal_structure(tile_kind)
    has_animal = tile_crop_or_animal >= 0

    # -1のまま引くと負インデックス(末尾参照)になるためクリップする。
    # 結果はapplies判定でマスクされるので誤って使われることはない。
    crop_idx = jnp.clip(tile_crop_or_animal, 0, C.N_CROPS - 1)
    animal_idx = jnp.clip(tile_crop_or_animal, 0, C.N_ANIMALS - 1)

    first_yield_day = _CROP_FIRST_YIELD_DAY[crop_idx]
    plant_matured = (day - tile_planted_or_placed_day) >= first_yield_day

    plant_applies = is_harvest_op & has_yield & is_plant & plant_matured
    animal_applies = is_harvest_op & has_yield & is_animal_structure & has_animal
    applies = plant_applies | animal_applies

    product_idx = jnp.where(
        plant_applies, _CROP_PRODUCT_IDX[crop_idx], _ANIMAL_PRODUCT_IDX[animal_idx]
    )
    units = jnp.where(applies, tile_yield_units, 0)
    new_inventory = (
        inventory + jax.nn.one_hot(product_idx, C.N_SHED_ITEMS, dtype=inventory.dtype) * units
    )

    new_yield_units = jnp.where(applies, 0, tile_yield_units)

    is_ongoing = _CROP_IS_ONGOING[crop_idx]
    removes_tile = plant_applies & jnp.logical_not(is_ongoing)
    new_tile_kind = jnp.where(removes_tile, C.TILE_EMPTY, tile_kind)
    new_tile_crop_or_animal = jnp.where(removes_tile, -1, tile_crop_or_animal)

    return new_tile_kind, new_tile_crop_or_animal, new_yield_units, new_inventory


def compute_plant_block(unit_ops, unit_crops, seeds):
    """1プレイヤーの全ユニット(farmer+hands)のPLANT要求を集計し、種切れの作物をブロックする。

    需要が在庫を超えた作物は、超過分だけでなく全要求がブロックされる(元コードの
    all-or-nothingルール)。

    Args:
        unit_ops: [N_UNITS] 各ユニットのop。
        unit_crops: [N_UNITS] 各ユニットがPLANTしようとしている作物インデックス
            (op != PLANTの要素は無視される)。
        seeds: [N_CROPS] このプレイヤーの残り種。

    Returns:
        [N_CROPS] bool。需要が在庫を超えた作物がTrue。
    """
    is_plant = unit_ops == C.FARMER_OP_PLANT
    demand_one_hot = jax.nn.one_hot(unit_crops, C.N_CROPS, dtype=jnp.int32)
    demand = jnp.sum(demand_one_hot * is_plant[:, None].astype(jnp.int32), axis=0)
    return demand > seeds


def apply_plant(
    op,
    crop_idx,
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
):
    """PLANT行動を1タイルに適用する(元コードの_new_plant)。

    blockedはcompute_plant_blockの結果を渡す。タイルが空(TILE_EMPTY)でなければ
    何もしない。

    Args:
        op: constants.FARMER_OP_*。
        crop_idx: 植えようとしている作物インデックス。
        blocked: [N_CROPS] bool。compute_plant_blockの結果。
        tile_kind: constants.TILE_*。
        tile_crop_or_animal: 現在の作物/動物インデックス。
        tile_planted_or_placed_day: 現在の植付/配置日。
        tile_watered_or_fed_today: 現在の水やり/餌やり済みフラグ。
        tile_consecutive_unwatered_or_unfed: 現在の未水やり/未餌やり連続日数。
        tile_yield_units: 現在の収穫可能量。
        tile_max_lifespan_step: 現在の寿命ステップ。
        tile_fertilized_until_day: 現在の施肥ボーナス最終日。
        day: 現在の日。
        turns_per_day: 1日のターン数。

    Returns:
        (new_tile_kind, new_tile_crop_or_animal, new_planted_day, new_watered_today,
        new_consecutive_unwatered, new_yield_units, new_max_lifespan_step,
        new_fertilized_until_day, seed_used)。
        seed_usedは[N_CROPS]で、実際に消費した種の作物だけ1。
    """
    is_plant_op = op == C.FARMER_OP_PLANT
    is_blocked = blocked[crop_idx]
    tile_empty = tile_kind == C.TILE_EMPTY
    applies = is_plant_op & jnp.logical_not(is_blocked) & tile_empty

    is_ongoing = _CROP_IS_ONGOING[crop_idx]
    max_yield_day = _CROP_MAX_YIELD_DAY[crop_idx]

    new_kind = jnp.where(applies, C.TILE_PLANT, tile_kind)
    new_crop = jnp.where(applies, crop_idx, tile_crop_or_animal)
    new_planted_day = jnp.where(applies, day, tile_planted_or_placed_day)
    new_watered_today = jnp.where(applies, False, tile_watered_or_fed_today)
    # 植付当日を未水やり1日目とする(元コード: consecutive_unwatered=1でスタート)
    new_consecutive_unwatered = jnp.where(applies, 1, tile_consecutive_unwatered_or_unfed)
    new_yield_units = jnp.where(applies, jnp.where(is_ongoing, 0, 1), tile_yield_units)
    new_max_lifespan_step = jnp.where(
        applies,
        jnp.where(is_ongoing, -1, (day + max_yield_day + 1) * turns_per_day),
        tile_max_lifespan_step,
    )
    new_fertilized_until_day = jnp.where(applies, -1, tile_fertilized_until_day)

    seed_used = jax.nn.one_hot(crop_idx, C.N_CROPS, dtype=jnp.int32) * applies.astype(jnp.int32)

    return (
        new_kind,
        new_crop,
        new_planted_day,
        new_watered_today,
        new_consecutive_unwatered,
        new_yield_units,
        new_max_lifespan_step,
        new_fertilized_until_day,
        seed_used,
    )


def apply_fertilize(op, tile_kind, tile_fertilized_until_day, day, inventory):
    """FERTILIZE行動を1タイルに適用する。

    対象がPLANTでない、または肥料を持っていなければ何もしない。ボーナスは
    day, day+1, day+2の3日間有効(元コード: fertilized_until_day = max(現在値, day+2))。

    Args:
        op: constants.FARMER_OP_*。
        tile_kind: constants.TILE_*。
        tile_fertilized_until_day: 現在の施肥ボーナス最終日。
        day: 現在の日。
        inventory: [N_SHED_ITEMS] このユニットの持ち物。

    Returns:
        (new_fertilized_until_day, new_inventory)。
    """
    is_fertilize_op = op == C.FARMER_OP_FERTILIZE
    is_plant = _is_plant(tile_kind)
    has_fertilizer = inventory[_FERTILIZER_IDX] > 0
    applies = is_fertilize_op & is_plant & has_fertilizer

    new_fertilized_until_day = jnp.where(
        applies, jnp.maximum(tile_fertilized_until_day, day + 2), tile_fertilized_until_day
    )
    new_inventory = inventory.at[_FERTILIZER_IDX].add(jnp.where(applies, -1, 0))

    return new_fertilized_until_day, new_inventory


def apply_dig(op, tile_kind, tile_crop_or_animal):
    """DIG行動を1タイルに適用する。

    PLANT/WEED、または動物のいない空のCOOP/PASTUREを更地に戻す。動物のいる
    COOP/PASTUREは対象外(掘れない)。yield_units等の他フィールドはリセットしない
    (kindがTILE_EMPTYになれば以降読まれないため。再度PLANTされた時点で上書きされる)。

    Args:
        op: constants.FARMER_OP_*。
        tile_kind: constants.TILE_*。
        tile_crop_or_animal: 現在の作物/動物インデックス(-1 = 該当なし)。

    Returns:
        (new_tile_kind, new_tile_crop_or_animal)。
    """
    is_dig_op = op == C.FARMER_OP_DIG
    is_plant_or_weed = (tile_kind == C.TILE_PLANT) | (tile_kind == C.TILE_WEED)
    is_empty_structure = _is_animal_structure(tile_kind) & (tile_crop_or_animal < 0)
    applies = is_dig_op & (is_plant_or_weed | is_empty_structure)

    new_tile_kind = jnp.where(applies, C.TILE_EMPTY, tile_kind)
    new_tile_crop_or_animal = jnp.where(applies, -1, tile_crop_or_animal)

    return new_tile_kind, new_tile_crop_or_animal
