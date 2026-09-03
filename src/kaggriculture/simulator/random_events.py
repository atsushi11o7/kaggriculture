"""雑草の自然発生・街の店の抽選(元コードの _spawn_weeds / _end_of_day末尾)。

JAXの乱数(threefry)とPythonのrandom.Random(メルセンヌ・ツイスタ)はアルゴリズムが
根本的に違うため、ビット単位で同じ乱数列を再現することはできない。ここでは
「確率分布として正しい」ことを目標にする(具体的にどのマスに雑草が生えるかは
元の実行結果と一致しなくてよい)。

state.State.rng_keyはエピソード開始時に決まる不変のベースキーで、日付ごとの
鍵はここでfold_inして毎回独立に導出する(前日までの消費状況を引き継がない)。
"""

import jax
import jax.numpy as jnp

from . import constants as C


def daily_rng_keys(rng_key, day):
    """その日専用のサブキーを3つ作る(player0の雑草用、player1の雑草用、店の抽選用)。

    Args:
        rng_key: エピソードのベースキー(State.rng_key)。
        day: 現在の日。

    Returns:
        [3, 2] のキー配列。[0]=player0の雑草用、[1]=player1の雑草用、[2]=店の抽選用。
    """
    day_key = jax.random.fold_in(rng_key, day)
    return jax.random.split(day_key, 3)


def spawn_weeds(key, tile_kind, weed_chance):
    """1プレイヤー分の盤面に雑草の自然発生を適用する(元コードの_spawn_weeds)。

    空き(TILE_EMPTY)のタイルだけが対象。

    Args:
        key: このプレイヤー・この日専用の乱数キー。
        tile_kind: [board, board] constants.TILE_*。
        weed_chance: 1マスあたりの雑草発生確率。

    Returns:
        new_tile_kind。
    """
    is_empty = tile_kind == C.TILE_EMPTY
    draws = jax.random.uniform(key, shape=tile_kind.shape)
    spawns = is_empty & (draws < weed_chance)
    return jnp.where(spawns, C.TILE_WEED, tile_kind)


def apply_shop_unlock(
    key, unlocked_shop_counts, next_day, shop_unlock_interval, max_shop_instances
):
    """街の店の抽選を適用する(元コードの_end_of_day末尾)。

    next_dayがshop_unlock_intervalの倍数の日にだけ、8種類の店から一様ランダムに
    1つ選んで出現数を+1する(重複可、max_shop_instancesで打ち止め)。

    Args:
        key: この日専用の乱数キー。
        unlocked_shop_counts: [N_SHOPS] 現在の店の種類ごとの出現数。
        next_day: 次の日(day + 1)。
        shop_unlock_interval: 何日おきに抽選するか。
        max_shop_instances: 出現数の合計上限。

    Returns:
        new_unlocked_shop_counts。
    """
    is_shop_day = (next_day > 0) & (next_day % shop_unlock_interval == 0)
    can_add = jnp.sum(unlocked_shop_counts) < max_shop_instances
    adds = is_shop_day & can_add

    drawn = jax.random.randint(key, (), 0, C.N_SHOPS)
    return jnp.where(adds, unlocked_shop_counts.at[drawn].add(1), unlocked_shop_counts)
