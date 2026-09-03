"""盤面の座標計算に関する共通ヘルパー(区画・納屋隣接マスなど)。

board_sizeは配列のshapeを決める静的な値として扱う。
"""

import jax.numpy as jnp


def quadrant_grid(board_size):
    """[board_size, board_size]。各マスの区画をconstants.QUADRANTSのインデックスで返す。"""
    half = board_size // 2
    ys, xs = jnp.meshgrid(jnp.arange(board_size), jnp.arange(board_size), indexing="ij")
    is_north = ys < half
    is_west = xs < half
    return jnp.where(is_north, jnp.where(is_west, 0, 1), jnp.where(is_west, 2, 3))


def shed_access_tiles(board_size):
    """[4, 2] 納屋隣接マスの座標(x, y)。NWSE順。"""
    half = board_size // 2
    return jnp.array(
        [[half - 1, half - 1], [half, half - 1], [half - 1, half], [half, half]], dtype=jnp.int32
    )


def is_shed_adjacent(pos, board_size):
    """納屋に隣接する4マス(中央の2x2)のどれかに立っているか。"""
    half = board_size // 2
    x, y = pos[0], pos[1]
    return ((x == half - 1) | (x == half)) & ((y == half - 1) | (y == half))


def default_spawn_position(board_size):
    """農夫・handの初期スポーン地点。NWにある納屋隣接マスをNWSE順で探す。

    board_sizeは静的値なので素のPythonループでよい。
    """
    half = board_size // 2
    for x, y in [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]:
        if y < half and x < half:
            return x, y
    return 0, 0
