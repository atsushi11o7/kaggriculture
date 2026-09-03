"""動物関連の行動(元コードの_apply_unit_action内、BUILD_*/PLACE/FEED/CARE/COLLECT_FERTILIZER)。

各関数は1ユニット・バッチ軸なしの純粋関数。バッチ適用時は呼び出し側でvmapする。
"""

import jax
import jax.numpy as jnp

from . import board
from . import constants as C
from . import game_params as P
from .constants import is_animal_structure as _is_animal_structure
from .inventory_actions import shed_room

_FERTILIZER_IDX = C.SHED_ITEMS.index("FERTILIZER")
_WHEAT_IDX = C.SHED_ITEMS.index("WHEAT")
_ANIMAL_STRUCTURE = jnp.array(P.ANIMAL_STRUCTURE, dtype=jnp.int32)


def apply_build(op, tile_kind):
    """BUILD_COOP/BUILD_PASTURE行動を1タイルに適用する。

    空(TILE_EMPTY)のタイルにのみ小屋を建てられる。tile_crop_or_animalは変化しない
    (EMPTYタイルは常に-1のため)。

    Args:
        op: constants.FARMER_OP_*。
        tile_kind: constants.TILE_*。

    Returns:
        new_tile_kind。
    """
    tile_empty = tile_kind == C.TILE_EMPTY
    builds_coop = (op == C.FARMER_OP_BUILD_COOP) & tile_empty
    builds_pasture = (op == C.FARMER_OP_BUILD_PASTURE) & tile_empty
    return jnp.where(builds_coop, C.TILE_COOP, jnp.where(builds_pasture, C.TILE_PASTURE, tile_kind))


def apply_feed(op, tile_kind, tile_crop_or_animal, tile_fed_today, inventory):
    """FEED行動を1タイルに適用する。

    動物のいるCOOP/PASTUREのみ対象。既に給餌済み、または小麦を持っていなければ
    何もしない。

    Args:
        op: constants.FARMER_OP_*。
        tile_kind: constants.TILE_*。
        tile_crop_or_animal: 現在の動物インデックス(-1 = 動物なし)。
        tile_fed_today: 既に給餌済みか。
        inventory: [N_SHED_ITEMS] このユニットの持ち物。

    Returns:
        (new_fed_today, new_inventory)。
    """
    is_feed_op = op == C.FARMER_OP_FEED
    is_animal_structure = _is_animal_structure(tile_kind)
    has_animal = tile_crop_or_animal >= 0
    not_fed_yet = jnp.logical_not(tile_fed_today)
    has_wheat = inventory[_WHEAT_IDX] > 0
    applies = is_feed_op & is_animal_structure & has_animal & not_fed_yet & has_wheat

    new_fed_today = jnp.where(applies, True, tile_fed_today)
    new_inventory = inventory.at[_WHEAT_IDX].add(jnp.where(applies, -1, 0))

    return new_fed_today, new_inventory


def apply_collect_fertilizer(
    op, tile_kind, tile_crop_or_animal, tile_fertilizer_available, inventory
):
    """COLLECT_FERTILIZER行動を1タイルに適用する。

    動物のいるCOOP/PASTUREで、未回収の肥料があれば1個回収する。

    Args:
        op: constants.FARMER_OP_*。
        tile_kind: constants.TILE_*。
        tile_crop_or_animal: 現在の動物インデックス(-1 = 動物なし)。
        tile_fertilizer_available: 未回収の肥料があるか。
        inventory: [N_SHED_ITEMS] このユニットの持ち物。

    Returns:
        (new_fertilizer_available, new_inventory)。
    """
    is_collect_op = op == C.FARMER_OP_COLLECT_FERTILIZER
    is_animal_structure = _is_animal_structure(tile_kind)
    has_animal = tile_crop_or_animal >= 0
    applies = is_collect_op & is_animal_structure & has_animal & tile_fertilizer_available

    new_fertilizer_available = jnp.where(applies, False, tile_fertilizer_available)
    new_inventory = inventory.at[_FERTILIZER_IDX].add(jnp.where(applies, 1, 0))

    return new_fertilizer_available, new_inventory


def apply_care(op, tile_kind, tile_crop_or_animal, tile_cared_today):
    """CARE行動を1タイルに適用する。

    動物のいるCOOP/PASTUREのみ対象。既に世話済みなら何もしない。

    Args:
        op: constants.FARMER_OP_*。
        tile_kind: constants.TILE_*。
        tile_crop_or_animal: 現在の動物インデックス(-1 = 動物なし)。
        tile_cared_today: 既に世話済みか。

    Returns:
        new_cared_today。
    """
    is_care_op = op == C.FARMER_OP_CARE
    is_animal_structure = _is_animal_structure(tile_kind)
    has_animal = tile_crop_or_animal >= 0
    not_cared_yet = jnp.logical_not(tile_cared_today)
    applies = is_care_op & is_animal_structure & has_animal & not_cared_yet

    return jnp.where(applies, True, tile_cared_today)


def apply_place(
    op,
    item_idx,
    n,
    pos,
    board_size,
    tile_kind,
    tile_crop_or_animal,
    tile_placed_day,
    tile_yield_units,
    tile_fed_today,
    tile_consecutive_unfed,
    tile_cared_today,
    tile_fertilizer_available,
    tile_pending_care_bonus,
    day,
    shed,
    inventory,
    shed_capacity,
):
    """PLACE行動を1ユニット分適用する。

    item_idxが動物で、今立っているタイルがその動物の空の小屋なら動物を配置する
    (元コードの_new_animal相当。持ち物にその動物が無ければ何も起きない)。
    それ以外は納屋隣接マスでの単一品目の納屋ドロップとして扱う(DROPと同様、
    容量を超える分は破棄)。動物配置の条件(品目が動物・小屋の種類が一致・
    小屋が空)を満たす場合、持ち物が足りなくても納屋ドロップにはフォールバック
    しない(元コード通り)。

    Args:
        op: constants.FARMER_OP_*。
        item_idx: 対象品目(constants.SHED_ITEMS)のインデックス。
        n: 納屋ドロップ時の要求数(動物配置では無視される)。
        pos: [2] (x, y) 現在位置。
        board_size: 盤面の一辺のサイズ。
        tile_kind: constants.TILE_*。
        tile_crop_or_animal: 現在の動物インデックス(-1 = 動物なし)。
        tile_placed_day: 現在の配置日。
        tile_yield_units: 現在の収穫可能量。
        tile_fed_today: 現在の給餌済みフラグ。
        tile_consecutive_unfed: 現在の未給餌連続日数。
        tile_cared_today: 現在の世話済みフラグ。
        tile_fertilizer_available: 現在の未回収肥料フラグ。
        tile_pending_care_bonus: 現在の世話ボーナス蓄積値。
        day: 現在の日。
        shed: [N_SHED_ITEMS] 納屋の在庫。
        inventory: [N_SHED_ITEMS] このユニットの持ち物。
        shed_capacity: 納屋の容量。

    Returns:
        (new_tile_crop_or_animal, new_tile_placed_day, new_tile_yield_units,
        new_tile_fed_today, new_tile_consecutive_unfed, new_tile_cared_today,
        new_tile_fertilizer_available, new_tile_pending_care_bonus,
        new_shed, new_inventory)。
    """
    is_place_op = op == C.FARMER_OP_PLACE
    is_animal_item = (item_idx >= C.N_PRODUCTS) & (item_idx < C.N_SHED_ITEMS)
    animal_idx = jnp.clip(item_idx - C.N_PRODUCTS, 0, C.N_ANIMALS - 1)
    structure_matches = tile_kind == _ANIMAL_STRUCTURE[animal_idx]
    structure_empty = tile_crop_or_animal < 0

    # 動物配置の「土俵に乗っているか」(持ち物の有無は問わない。元コードはここが
    # 真なら持ち物不足でも納屋ドロップにフォールバックしないため)
    on_animal_branch = is_place_op & is_animal_item & structure_matches & structure_empty
    places_animal = on_animal_branch & (inventory[item_idx] > 0)

    new_crop_or_animal = jnp.where(places_animal, animal_idx, tile_crop_or_animal)
    new_placed_day = jnp.where(places_animal, day, tile_placed_day)
    new_yield_units = jnp.where(places_animal, 0, tile_yield_units)
    new_fed_today = jnp.where(places_animal, False, tile_fed_today)
    new_consecutive_unfed = jnp.where(places_animal, 0, tile_consecutive_unfed)
    new_cared_today = jnp.where(places_animal, False, tile_cared_today)
    new_fertilizer_available = jnp.where(places_animal, False, tile_fertilizer_available)
    new_pending_care_bonus = jnp.where(places_animal, 0, tile_pending_care_bonus)

    one_hot = jax.nn.one_hot(item_idx, C.N_SHED_ITEMS, dtype=jnp.int32)
    inventory_after_place = inventory - one_hot * places_animal.astype(jnp.int32)

    # 納屋ドロップ側(動物配置の土俵に乗っていない場合のみ)
    is_shed_drop = (
        is_place_op & jnp.logical_not(on_animal_branch) & board.is_shed_adjacent(pos, board_size)
    )
    n_take = jnp.minimum(jnp.maximum(n, 0), inventory_after_place[item_idx])
    n_take = jnp.minimum(n_take, shed_room(shed, shed_capacity))
    n_take = jnp.where(is_shed_drop & (n_take > 0), n_take, 0)

    new_shed = shed + one_hot * n_take
    new_inventory = inventory_after_place - one_hot * n_take

    return (
        new_crop_or_animal,
        new_placed_day,
        new_yield_units,
        new_fed_today,
        new_consecutive_unfed,
        new_cared_today,
        new_fertilizer_available,
        new_pending_care_bonus,
        new_shed,
        new_inventory,
    )
