"""日末処理のうち、日次更新・雑草発生以外の残り(元コードの_end_of_day)。

farmer/hands全員の持ち物を納屋へ払い出し、位置・雇用状態を翌日用にリセットする。
"""

import jax
import jax.numpy as jnp

from . import board
from . import constants as C
from .inventory_actions import dump_inventory


def drop_all_inventories(farmer_inventory, hands_inventory, shed, shed_capacity):
    """farmer+全handの持ち物を、この順序で納屋へ払い出す(元コードの_drop_inventories_to_shed)。

    容量を超える分は破棄される。farmer→hand0→hand1…の順で処理するため、
    先に処理される方が納屋の残り容量を優先的に使う。未雇用のhandの持ち物は
    常に0なので、実際にどれだけ払い出すかには影響しない。

    Args:
        farmer_inventory: [N_SHED_ITEMS]。
        hands_inventory: [MAX_HANDS, N_SHED_ITEMS]。
        shed: [N_SHED_ITEMS] 納屋の在庫。
        shed_capacity: 納屋の容量。

    Returns:
        new_shed。
    """
    all_inventory = jnp.concatenate([farmer_inventory[None], hands_inventory], axis=0)

    def step(shed, inv):
        new_shed, _ = dump_inventory(shed, inv, shed_capacity)
        return new_shed, None

    final_shed, _ = jax.lax.scan(step, shed, all_inventory)
    return final_shed


def reset_units_for_new_day(board_size):
    """farmer/handsの位置・雇用状態・持ち物を翌日用にリセットする。

    元コードのfarm["farmer"]=_default_spawn(...), farm["hands"]=[],
    farm["hires_today"]=0, private["inventories"]=[{}] に対応。

    Args:
        board_size: 盤面の一辺のサイズ。

    Returns:
        (farmer_pos, hands_pos, hands_active, hires_today, farmer_inventory, hands_inventory)。
    """
    spawn_x, spawn_y = board.default_spawn_position(board_size)
    farmer_pos = jnp.array([spawn_x, spawn_y], dtype=jnp.int32)
    hands_pos = jnp.zeros((C.MAX_HANDS, 2), dtype=jnp.int32)
    hands_active = jnp.zeros((C.MAX_HANDS,), dtype=bool)
    hires_today = jnp.array(0, dtype=jnp.int32)
    farmer_inventory = jnp.zeros((C.N_SHED_ITEMS,), dtype=jnp.int32)
    hands_inventory = jnp.zeros((C.MAX_HANDS, C.N_SHED_ITEMS), dtype=jnp.int32)

    return farmer_pos, hands_pos, hands_active, hires_today, farmer_inventory, hands_inventory
