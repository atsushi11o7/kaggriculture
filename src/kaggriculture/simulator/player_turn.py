"""1プレイヤー分のユニット行動(farmer + hands)を1ターン適用する。

元コードのinterpreter内、farmer→hand0→hand1…の順に処理するループに対応。
納屋(shed)は全ユニットで共有されるため、jax.lax.while_loopで真に逐次処理する
(vmapでは表現できない)。種(seeds)はcompute_plant_blockが「需要<=在庫」を
既に保証しているので、逐次ではなく全ユニット分の消費をまとめて1回引く。

ループ回数は「farmer + 実際に雇われているhand数」までで打ち切る(MAX_HANDS=32
分を毎回処理しない)。hands_activeは常に配列の先頭から連続してTrue(apply_hireが
最初の空きスロットを埋め、日中の個別解雇も無いため)なので、これで安全に打ち切れる。
未処理のまま残るhandスロットは入力の値をそのまま返す(PASS相当なので不変)。
"""

import jax
import jax.numpy as jnp

from . import constants as C
from .crop_actions import compute_plant_block
from .unit_actions import apply_unit_action


def apply_player_units(
    farmer_op,
    farmer_arg_idx,
    farmer_n,
    hands_op,
    hands_arg_idx,
    hands_n,
    hands_active,
    farmer_pos,
    hands_pos,
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
    farmer_inventory,
    hands_inventory,
    seeds,
    shed,
    board_size,
    day,
    turns_per_day,
    shed_capacity,
):
    """1プレイヤーのfarmer+hands全員の行動を、この順序で逐次適用する。

    Args:
        farmer_op, farmer_arg_idx, farmer_n: farmerの行動。
        hands_op, hands_arg_idx, hands_n: [MAX_HANDS] 各handの行動。
        hands_active: [MAX_HANDS] bool。そのhandが今日雇われているか。
        farmer_pos: [2] farmerの現在位置。
        hands_pos: [MAX_HANDS, 2] 各handの現在位置。
        tile_*: [board, board] 盤面の各チャンネル(11個)。
        farmer_inventory: [N_SHED_ITEMS] farmerの持ち物。
        hands_inventory: [MAX_HANDS, N_SHED_ITEMS] 各handの持ち物。
        seeds: [N_CROPS] このプレイヤーの残り種。
        shed: [N_SHED_ITEMS] 納屋の在庫。
        board_size: 盤面の一辺のサイズ。
        day: 現在の日。
        turns_per_day: 1日のターン数。
        shed_capacity: 納屋の容量。

    Returns:
        (new_farmer_pos, new_hands_pos, new_tile_* ×11, new_farmer_inventory,
        new_hands_inventory, new_seeds, new_shed)。
    """
    # 未雇用のhandは常にPASS扱い(PLANT需要に加算されず、行動もしない)
    hands_op_eff = jnp.where(hands_active, hands_op, C.FARMER_OP_PASS)

    unit_ops = jnp.concatenate([farmer_op[None], hands_op_eff])
    unit_arg_idx = jnp.concatenate([farmer_arg_idx[None], hands_arg_idx])
    unit_n = jnp.concatenate([farmer_n[None], hands_n])
    unit_pos = jnp.concatenate([farmer_pos[None], hands_pos], axis=0)
    unit_inventory = jnp.concatenate([farmer_inventory[None], hands_inventory], axis=0)

    blocked = compute_plant_block(unit_ops, unit_arg_idx, seeds)

    tile_fields_init = (
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
    )

    # farmer(1) + 実際に雇われているhand数まで(モジュールdocstring参照)。
    n_units = 1 + jnp.sum(hands_active.astype(jnp.int32))

    def cond(carry):
        *_, i = carry
        return i < n_units

    def step_unit(carry):
        tile_fields, shed, all_pos, all_inv, all_seed_used, i = carry
        pos = unit_pos[i]
        op = unit_ops[i]
        arg_idx = unit_arg_idx[i]
        n = unit_n[i]
        inv = unit_inventory[i]
        xi, yi = pos[0], pos[1]

        gathered = tuple(f[yi, xi] for f in tile_fields)

        out = apply_unit_action(
            op,
            arg_idx,
            n,
            pos,
            board_size,
            *gathered,
            day,
            turns_per_day,
            shed,
            inv,
            shed_capacity,
            blocked,
        )
        new_pos, new_tile_values, new_shed, new_inv, seed_used = (
            out[0],
            out[1:12],
            out[12],
            out[13],
            out[14],
        )

        new_tile_fields = tuple(
            f.at[yi, xi].set(v) for f, v in zip(tile_fields, new_tile_values, strict=True)
        )
        all_pos = all_pos.at[i].set(new_pos)
        all_inv = all_inv.at[i].set(new_inv)
        all_seed_used = all_seed_used.at[i].set(seed_used)

        return new_tile_fields, new_shed, all_pos, all_inv, all_seed_used, i + 1

    # 未処理スロット(雇われていないhand)の初期値は入力そのもの(PASS相当なので
    # 処理してもしなくても同じ値になる)。seed_usedだけは「消費した種」なので0初期化。
    init_all_seed_used = jnp.zeros((C.MAX_HANDS + 1, C.N_CROPS), dtype=jnp.int32)

    (final_tile_fields, final_shed, all_pos, all_inv, all_seed_used, _) = jax.lax.while_loop(
        cond,
        step_unit,
        (tile_fields_init, shed, unit_pos, unit_inventory, init_all_seed_used, 0),
    )

    new_seeds = seeds - jnp.sum(all_seed_used, axis=0)

    return (
        all_pos[0],
        all_pos[1:],
        *final_tile_fields,
        all_inv[0],
        all_inv[1:],
        new_seeds,
        final_shed,
    )
