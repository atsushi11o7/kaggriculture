"""batch_size局分の初期状態(State)を作る。

元コードの _initialize / _new_farm / _new_private / _new_market / _new_town に対応。

board_size は配列のshapeを決めるので、jax.jitでコンパイルする際は
静的引数(static_argnums)として扱う必要がある(実行時に変わる値ではなく、
コンパイル前に決まっている定数として扱う)。
"""

import jax
import jax.numpy as jnp

from . import board
from . import constants as C
from . import game_params as P
from .state import State


def reset(key, batch_size, board_size=10, starting_money=3000.0):
    """batch_size局分の初期Stateを作る。

    key: jax.random.PRNGKey。ここからbatch_size個に分割し、各局のベースシード
         (State.rng_key)にする。
    """
    tile_shape = (batch_size, 2, board_size, board_size)

    # NWだけ解放済み(TILE_EMPTY)、他はTILE_LOCKED
    quadrant_grid = board.quadrant_grid(board_size)  # [board, board]
    tiles_kind_2d = jnp.where(quadrant_grid == 0, C.TILE_EMPTY, C.TILE_LOCKED)
    tiles_kind = jnp.broadcast_to(tiles_kind_2d, tile_shape)

    zeros_i32 = jnp.zeros(tile_shape, dtype=jnp.int32)
    zeros_bool = jnp.zeros(tile_shape, dtype=bool)

    spawn_x, spawn_y = board.default_spawn_position(board_size)
    farmer_pos = jnp.broadcast_to(
        jnp.array([spawn_x, spawn_y], dtype=jnp.int32), (batch_size, 2, 2)
    )

    unlocked_quadrants = jnp.zeros((batch_size, 2, C.N_QUADRANTS), dtype=bool)
    unlocked_quadrants = unlocked_quadrants.at[:, :, 0].set(True)  # NW(インデックス0)のみ解放

    return State(
        # --- 盤面 ---
        tiles_kind=tiles_kind,
        tiles_crop_or_animal=jnp.full(tile_shape, -1, dtype=jnp.int32),  # 何も植わっていない
        tiles_planted_or_placed_day=zeros_i32,
        tiles_watered_or_fed_today=zeros_bool,
        tiles_consecutive_unwatered_or_unfed=zeros_i32,
        tiles_yield_units=zeros_i32,
        tiles_max_lifespan_step=jnp.full(tile_shape, -1, dtype=jnp.int32),
        tiles_fertilized_until_day=jnp.full(tile_shape, -1, dtype=jnp.int32),
        tiles_cared_today=zeros_bool,
        tiles_fertilizer_available=zeros_bool,
        tiles_pending_care_bonus=zeros_i32,
        # --- ユニット位置 ---
        farmer_pos=farmer_pos,
        hands_pos=jnp.zeros((batch_size, 2, C.MAX_HANDS, 2), dtype=jnp.int32),
        hands_active=jnp.zeros((batch_size, 2, C.MAX_HANDS), dtype=bool),  # 初日は0人
        hires_today=jnp.zeros((batch_size, 2), dtype=jnp.int32),
        # --- 農場全体 ---
        money=jnp.full((batch_size, 2), starting_money, dtype=jnp.float32),
        unlocked_quadrants=unlocked_quadrants,
        # --- 非公開状態 ---
        shed=jnp.zeros((batch_size, 2, C.N_SHED_ITEMS), dtype=jnp.int32),
        seeds=jnp.zeros((batch_size, 2, C.N_CROPS), dtype=jnp.int32),
        farmer_inventory=jnp.zeros((batch_size, 2, C.N_SHED_ITEMS), dtype=jnp.int32),
        hands_inventory=jnp.zeros((batch_size, 2, C.MAX_HANDS, C.N_SHED_ITEMS), dtype=jnp.int32),
        # --- 市場・街(共有) ---
        market_inventory=jnp.full((batch_size, C.N_PRODUCTS), P.MARKET_I0, dtype=jnp.int32),
        town_shop_counts=jnp.zeros((batch_size, C.N_SHOPS), dtype=jnp.int32),
        # --- 時間・乱数 ---
        step=jnp.zeros((batch_size,), dtype=jnp.int32),
        rng_key=jax.random.split(key, batch_size),
    )
