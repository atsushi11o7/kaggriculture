"""市場注文のうち、ロックステップループ不要な4種類(HIRE/BUY_LAND/BUY_SEED/BUY_ANIMAL)。

HIRE/BUY_LANDは元コードの_do_hire/_do_buy_landに対応し、キュー内の同じ位置
(スロット)の注文をプレイヤー順に1回ずつ処理する。BUY_SEED/BUY_ANIMALは価格が
固定で市場在庫と無関係なため、SELL/BUY_PRODUCTのようなロックステップループが
不要で閉じた式で計算できる(market_lockstep.py参照)。
"""

import jax
import jax.numpy as jnp

from . import board
from . import constants as C
from . import game_params as P


def _build_fib_table(length):
    """fib(0)=1, fib(1)=1, fib(2)=2, ... をlength個並べたテーブル。"""
    table = []
    a, b = 1, 1
    for _ in range(length):
        table.append(a)
        a, b = b, a + b
    return table


# hires_todayで直接引けるようテーブル化する。hires_todayはMAX_HANDS(雇えるhandの上限)
# を超えないので、この長さで十分。
_FIB_TABLE = jnp.array(_build_fib_table(C.MAX_HANDS + 1), dtype=jnp.int32)
_LAND_PRICES = jnp.array(P.LAND_PRICES, dtype=jnp.float32)
_CROP_SEED_COST = jnp.array(P.CROP_SEED_COST, dtype=jnp.float32)
_ANIMAL_COST = jnp.array(P.ANIMAL_COST, dtype=jnp.float32)


def apply_hire(op, money, hires_today, hands_active, hands_pos, farmer_pos, board_size, hire_mult):
    """HIRE注文を適用する(元コードの_do_hire)。

    コストはfib(hires_today)*hire_mult。空いているhandスロットが無い場合
    (元のルールには無いMAX_HANDS上限に到達した場合)は、所持金不足の時と同様に
    何もしない。

    Args:
        op: constants.MARKET_OP_*。
        money: このプレイヤーの所持金。
        hires_today: 今日すでに雇った人数。
        hands_active: [MAX_HANDS] bool。
        hands_pos: [MAX_HANDS, 2]。
        farmer_pos: [2]。
        board_size: 盤面の一辺のサイズ。
        hire_mult: farmHandCostMult。

    Returns:
        (new_money, new_hires_today, new_hands_active, new_hands_pos)。
    """
    cost = _FIB_TABLE[jnp.clip(hires_today, 0, C.MAX_HANDS)] * hire_mult
    free_slot_idx = jnp.argmin(
        hands_active.astype(jnp.int32)
    )  # 最初の空きスロット(全て埋まっていれば0)
    has_free_slot = jnp.logical_not(jnp.all(hands_active))
    applies = (op == C.MARKET_OP_HIRE) & has_free_slot & (money >= cost)

    shed_tiles = board.shed_access_tiles(board_size)  # [4, 2]
    all_pos = jnp.concatenate([farmer_pos[None], hands_pos], axis=0)
    all_active = jnp.concatenate([jnp.array([True]), hands_active])
    occupancy = jnp.sum(
        jnp.all(all_pos[:, None, :] == shed_tiles[None, :, :], axis=-1) & all_active[:, None],
        axis=0,
    )
    spawn_pos = shed_tiles[jnp.argmin(occupancy)]

    new_money = jnp.where(applies, money - cost, money)
    new_hires_today = jnp.where(applies, hires_today + 1, hires_today)
    new_hands_active = jnp.where(applies, hands_active.at[free_slot_idx].set(True), hands_active)
    new_hands_pos = jnp.where(applies, hands_pos.at[free_slot_idx].set(spawn_pos), hands_pos)

    return new_money, new_hires_today, new_hands_active, new_hands_pos


def apply_buy_land(op, money, unlocked_quadrants, tile_kind, board_size):
    """BUY_LAND注文を適用する(元コードの_do_buy_land)。

    NE→SW→SEの順に$1,000/$2,000/$4,000で解放する。全区画解放済みなら何もしない。

    Args:
        op: constants.MARKET_OP_*。
        money: このプレイヤーの所持金。
        unlocked_quadrants: [N_QUADRANTS] bool(constants.QUADRANTS順)。
        tile_kind: [board, board] constants.TILE_*。
        board_size: 盤面の一辺のサイズ。

    Returns:
        (new_money, new_unlocked_quadrants, new_tile_kind)。
    """
    n_unlocked_extra = jnp.sum(unlocked_quadrants.astype(jnp.int32)) - 1  # NW分を除く
    all_bought = n_unlocked_extra >= (C.N_QUADRANTS - 1)
    idx = jnp.clip(n_unlocked_extra, 0, C.N_QUADRANTS - 2)
    cost = _LAND_PRICES[idx]
    applies = (op == C.MARKET_OP_BUY_LAND) & jnp.logical_not(all_bought) & (money >= cost)

    # QUADRANTS=("NW","NE","SW","SE")なので、n_unlocked_extra+1番目の区画を解放する
    target_quadrant = idx + 1

    new_money = jnp.where(applies, money - cost, money)
    new_unlocked_quadrants = jnp.where(
        applies, unlocked_quadrants.at[target_quadrant].set(True), unlocked_quadrants
    )

    quadrant_grid = board.quadrant_grid(board_size)
    unlocks_here = applies & (quadrant_grid == target_quadrant) & (tile_kind == C.TILE_LOCKED)
    new_tile_kind = jnp.where(unlocks_here, C.TILE_EMPTY, tile_kind)

    return new_money, new_unlocked_quadrants, new_tile_kind


def apply_buy_seed(op, item_idx, n, money, seeds):
    """BUY_SEED注文を適用する(元コードの_commit_unitのBUY_SEED分岐)。

    種の価格は市場在庫と無関係な固定値で、購入数による変動も無いため、1個ずつ
    ではなく閉じた式で「実際に買える数」を計算する(n・所持金がどれだけ大きくても
    正確。SELL/BUY_PRODUCT/BUY_ANIMALと違ってロックステップループが不要)。

    Args:
        op: constants.MARKET_OP_*。
        item_idx: 対象作物(constants.CROPS)のインデックス。
        n: 要求数。
        money: このプレイヤーの所持金。
        seeds: [N_CROPS] このプレイヤーの残り種。

    Returns:
        (new_money, new_seeds)。
    """
    price = _CROP_SEED_COST[item_idx]
    affordable = jnp.floor(money / price).astype(jnp.int32)
    actual = jnp.where(op == C.MARKET_OP_BUY_SEED, jnp.minimum(jnp.maximum(n, 0), affordable), 0)

    new_money = money - actual.astype(jnp.float32) * price
    new_seeds = seeds + jax.nn.one_hot(item_idx, C.N_CROPS, dtype=jnp.int32) * actual

    return new_money, new_seeds


def apply_buy_animal(op, animal_idx, n, money, shed, shed_capacity):
    """BUY_ANIMAL注文を適用する(元コードの_commit_unitのBUY_ANIMAL分岐)。

    価格は固定(市場在庫と無関係)。制約は所持金と納屋の残り容量の2つだけで、
    どちらも購入数に対して単純に効くだけ(価格が変動しない)なので、BUY_SEEDと
    同様に閉じた式で計算できる。

    Args:
        op: constants.MARKET_OP_*。
        animal_idx: 対象動物(constants.ANIMALS)のインデックス。
        n: 要求数。
        money: このプレイヤーの所持金。
        shed: [N_SHED_ITEMS] 納屋の在庫。
        shed_capacity: 納屋の容量。

    Returns:
        (new_money, new_shed)。
    """
    item_idx = C.N_PRODUCTS + animal_idx  # SHED_ITEMS = PRODUCTS + ANIMALS
    price = _ANIMAL_COST[animal_idx]
    affordable = jnp.floor(money / price).astype(jnp.int32)
    room = jnp.maximum(0, shed_capacity - jnp.sum(shed))
    actual = jnp.where(
        op == C.MARKET_OP_BUY_ANIMAL,
        jnp.minimum(jnp.maximum(n, 0), jnp.minimum(affordable, room)),
        0,
    )

    new_money = money - actual.astype(jnp.float32) * price
    new_shed = shed + jax.nn.one_hot(item_idx, C.N_SHED_ITEMS, dtype=jnp.int32) * actual

    return new_money, new_shed
