"""持ち物・納屋関連の行動(元コードの_apply_unit_action内、PICKUP/DROP、および日末の強制払い出し)。

各関数は1ユニット・バッチ軸なしの純粋関数。バッチ適用時は呼び出し側でvmapする。
"""

import jax
import jax.numpy as jnp

from . import board
from . import constants as C


def shed_room(shed, shed_capacity):
    """納屋の残り容量。"""
    return jnp.maximum(0, shed_capacity - jnp.sum(shed))


def apply_pickup(op, item_idx, n, pos, board_size, shed, inventory):
    """PICKUP行動を1ユニット分適用する。

    納屋隣接マスでのみ機能。指定品目を最大n個、納屋からこのユニットの持ち物へ移す
    (在庫が無ければ可能な分だけ)。

    Args:
        op: constants.FARMER_OP_*。
        item_idx: 対象品目(constants.SHED_ITEMS)のインデックス。
        n: 要求数。
        pos: [2] (x, y) 現在位置。
        board_size: 盤面の一辺のサイズ。
        shed: [N_SHED_ITEMS] 納屋の在庫。
        inventory: [N_SHED_ITEMS] このユニットの持ち物。

    Returns:
        (new_shed, new_inventory)。
    """
    adjacent = board.is_shed_adjacent(pos, board_size)
    take = jnp.minimum(jnp.maximum(n, 0), shed[item_idx])
    applies = (op == C.FARMER_OP_PICKUP) & adjacent & (take > 0)
    take = jnp.where(applies, take, 0)

    one_hot = jax.nn.one_hot(item_idx, C.N_SHED_ITEMS, dtype=jnp.int32)
    return shed - one_hot * take, inventory + one_hot * take


def apply_drop(op, pos, board_size, shed, inventory, shed_capacity):
    """DROP行動を1ユニット分適用する。

    納屋隣接マスで、持ち物を全て納屋に移す(容量を超える分は破棄)。複数品目を
    同時に持っている場合、元コードは「拾った順」で処理するが、テンソル版は
    constants.SHED_ITEMSの並び順で処理する(納屋がほぼ満杯で複数品目を同時に
    持っている稀なケースのみ挙動に差が出る)。

    Args:
        op: constants.FARMER_OP_*。
        pos: [2] (x, y) 現在位置。
        board_size: 盤面の一辺のサイズ。
        shed: [N_SHED_ITEMS] 納屋の在庫。
        inventory: [N_SHED_ITEMS] このユニットの持ち物。
        shed_capacity: 納屋の容量。

    Returns:
        (new_shed, new_inventory)。
    """
    applies = (op == C.FARMER_OP_DROP) & board.is_shed_adjacent(pos, board_size)
    new_shed, new_inventory = dump_inventory(shed, inventory, shed_capacity)

    return jnp.where(applies, new_shed, shed), jnp.where(applies, new_inventory, inventory)


def dump_inventory(shed, inventory, shed_capacity):
    """1ユニット分の持ち物を無条件で全て納屋に移す(容量を超える分は破棄)。

    apply_drop(プレイヤーのDROP行動)と、日末の強制払い出し(end_of_day.py)の
    両方から使われる共通処理。

    Args:
        shed: [N_SHED_ITEMS] 納屋の在庫。
        inventory: [N_SHED_ITEMS] このユニットの持ち物。
        shed_capacity: 納屋の容量。

    Returns:
        (new_shed, new_inventory)。
    """

    def body(i, carry):
        shed, inv = carry
        take = jnp.minimum(inv[i], shed_room(shed, shed_capacity))
        return shed.at[i].add(take), inv.at[i].set(0)

    return jax.lax.fori_loop(0, C.N_SHED_ITEMS, body, (shed, inventory))
