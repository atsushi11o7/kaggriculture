"""作物・動物の日次更新(元コードの _daily_refresh_plants / _daily_refresh_animals)。

タイルごとに独立した要素ごとの処理なので、盤面全体の配列にそのまま適用できる
(vmap不要)。乱数は使わない(雑草の自然発生・店の抽選は別モジュール)。
"""

import jax.numpy as jnp

from . import constants as C
from . import game_params as P
from .constants import is_animal_structure as _is_animal_structure
from .constants import is_plant as _is_plant
from .game_params import CROP_FIRST_YIELD_DAY_JAX as _CROP_FIRST_YIELD_DAY
from .game_params import CROP_IS_ONGOING_JAX as _CROP_IS_ONGOING
from .game_params import CROP_MAX_YIELD_JAX as _CROP_MAX_YIELD

_CROP_INTERVAL = jnp.array(P.CROP_INTERVAL, dtype=jnp.int32)

_ANIMAL_FIRST_YIELD_DAY = jnp.array(P.ANIMAL_FIRST_YIELD_DAY, dtype=jnp.int32)
_ANIMAL_INTERVAL = jnp.array(P.ANIMAL_INTERVAL, dtype=jnp.int32)
_ANIMAL_MAX_HELD = jnp.array(P.ANIMAL_MAX_HELD, dtype=jnp.int32)


def apply_daily_refresh_plants(
    tile_kind,
    tile_crop,
    tile_planted_day,
    tile_watered_today,
    tile_consecutive_unwatered,
    tile_yield_units,
    tile_max_lifespan_step,
    tile_fertilized_until_day,
    current_day,
    turns_per_day,
):
    """作物の日次更新を盤面全体に適用する。

    未水やり連続日数を更新し、2日連続ならWEED化する。継続収穫型のみ、
    スケジュール通りの生産日ならyield_unitsを増やす(施肥+水やり済みなら+2、
    そうでなければ+1)。生産回数が上限に達したら寿命ステップを設定する。

    Args:
        tile_kind: constants.TILE_*。
        tile_crop: 作物インデックス。
        tile_planted_day: 植えた日。
        tile_watered_today: 今日水やり済みか(この関数でFalseにリセットされる)。
        tile_consecutive_unwatered: 未水やり連続日数。
        tile_yield_units: 現在の収穫可能量。
        tile_max_lifespan_step: 現在の寿命ステップ。
        tile_fertilized_until_day: 施肥ボーナス最終日。
        current_day: 更新前の日(=今終わろうとしている日)。
        turns_per_day: 1日のターン数。

    Returns:
        (new_tile_kind, new_tile_crop, new_watered_today, new_consecutive_unwatered,
        new_yield_units, new_max_lifespan_step)。
    """
    is_plant = _is_plant(tile_kind)
    was_watered = tile_watered_today

    new_consecutive_unwatered = jnp.where(
        is_plant,
        jnp.where(was_watered, 0, tile_consecutive_unwatered + 1),
        tile_consecutive_unwatered,
    )
    new_watered_today = jnp.where(is_plant, False, tile_watered_today)
    becomes_weed = is_plant & (new_consecutive_unwatered >= 2)

    crop_idx = jnp.clip(tile_crop, 0, C.N_CROPS - 1)
    is_ongoing = _CROP_IS_ONGOING[crop_idx]
    first_yield_day = _CROP_FIRST_YIELD_DAY[crop_idx]
    interval = _CROP_INTERVAL[crop_idx]
    max_yield = _CROP_MAX_YIELD[crop_idx]

    next_day = current_day + 1
    days_since_first = next_day - tile_planted_day - first_yield_day
    is_production_tick = (days_since_first >= 0) & (
        days_since_first % jnp.maximum(interval, 1) == 0
    )
    production_count = days_since_first // jnp.maximum(interval, 1) + 1
    within_cap = production_count <= max_yield

    fertilized = was_watered & (tile_fertilized_until_day >= current_day)
    bonus = jnp.where(fertilized, 2, 1)

    produces = (
        is_plant & jnp.logical_not(becomes_weed) & is_ongoing & is_production_tick & within_cap
    )
    new_yield_units = jnp.where(
        produces, jnp.minimum(max_yield, tile_yield_units + bonus), tile_yield_units
    )

    reaches_cap = produces & (production_count == max_yield)
    new_max_lifespan_step = jnp.where(
        reaches_cap, (next_day + 1) * turns_per_day, tile_max_lifespan_step
    )

    new_tile_kind = jnp.where(becomes_weed, C.TILE_WEED, tile_kind)
    new_tile_crop = jnp.where(becomes_weed, -1, tile_crop)

    return (
        new_tile_kind,
        new_tile_crop,
        new_watered_today,
        new_consecutive_unwatered,
        new_yield_units,
        new_max_lifespan_step,
    )


def apply_daily_refresh_animals(
    tile_kind,
    tile_animal,
    tile_placed_day,
    tile_fed_today,
    tile_consecutive_unfed,
    tile_yield_units,
    tile_cared_today,
    tile_fertilizer_available,
    tile_pending_care_bonus,
    day,
):
    """動物の日次更新を盤面全体に適用する。

    未給餌連続日数を更新し、2日連続なら逃げる(小屋は残る)。スケジュール通りの
    生産日なら、給餌済みの場合に限り蓄積したCAREボーナスを丸ごと払い出して
    yield_unitsに加算し、蓄積値を0に戻す(未給餌なら基礎の1だけ生産され、
    蓄積値は払い出されずに0にリセットされる)。世話も給餌も済んでいれば、
    (生産日かどうかに関わらず)蓄積値へさらに+1する。元コードはこの払い出しと
    蓄積が2つの独立したif文で順に実行されるため、生産日かつ世話+給餌済みの日は
    「払い出して0にリセットしてから、直後に+1される」形になる。

    Args:
        tile_kind: constants.TILE_*。
        tile_animal: 動物インデックス(-1 = 動物なし)。
        tile_placed_day: 配置した日。
        tile_fed_today: 今日給餌済みか(この関数でFalseにリセットされる)。
        tile_consecutive_unfed: 未給餌連続日数。
        tile_yield_units: 現在の収穫可能量。
        tile_cared_today: 今日世話済みか(この関数でFalseにリセットされる)。
        tile_fertilizer_available: 未回収の肥料フラグ。
        tile_pending_care_bonus: 蓄積されたCAREボーナス。
        day: 更新前の日。

    Returns:
        (new_tile_animal, new_fed_today, new_consecutive_unfed, new_yield_units,
        new_cared_today, new_fertilizer_available, new_pending_care_bonus)。
    """
    is_animal = _is_animal_structure(tile_kind) & (tile_animal >= 0)

    new_consecutive_unfed = jnp.where(
        is_animal, jnp.where(tile_fed_today, 0, tile_consecutive_unfed + 1), tile_consecutive_unfed
    )
    escapes = is_animal & (new_consecutive_unfed >= 2)
    survives = is_animal & jnp.logical_not(escapes)

    animal_idx = jnp.clip(tile_animal, 0, C.N_ANIMALS - 1)
    first_yield_day = _ANIMAL_FIRST_YIELD_DAY[animal_idx]
    interval = _ANIMAL_INTERVAL[animal_idx]
    max_held = _ANIMAL_MAX_HELD[animal_idx]

    next_day = day + 1
    days_since_first = next_day - tile_placed_day - first_yield_day
    is_production_tick = (days_since_first >= 0) & (
        days_since_first % jnp.maximum(interval, 1) == 0
    )
    produces = survives & is_production_tick

    paid_bonus = jnp.where(tile_fed_today, tile_pending_care_bonus, 0)
    new_yield_units = jnp.where(
        produces, jnp.minimum(max_held, tile_yield_units + 1 + paid_bonus), tile_yield_units
    )

    # 生産ティックならまず0にリセット(払い出し済み)、その後cared+fedなら+1
    after_production = jnp.where(produces, 0, tile_pending_care_bonus)
    banks_bonus = survives & tile_cared_today & tile_fed_today
    new_pending_care_bonus = jnp.where(banks_bonus, after_production + 1, after_production)

    new_fertilizer_available = jnp.where(survives, True, tile_fertilizer_available)
    new_fed_today = jnp.where(is_animal, False, tile_fed_today)
    new_cared_today = jnp.where(is_animal, False, tile_cared_today)
    new_tile_animal = jnp.where(escapes, -1, tile_animal)

    return (
        new_tile_animal,
        new_fed_today,
        new_consecutive_unfed,
        new_yield_units,
        new_cared_today,
        new_fertilizer_available,
        new_pending_care_bonus,
    )
