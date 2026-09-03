"""Stateから、各プレイヤーが実際に見てよい観測を作る(相手の非公開情報を隠す)。

Kaggle提供版のinterpreterは、各プレイヤーに`private`(納屋・種・持ち物)を隠した
observationを渡す(それ以外のtiles/money/位置等は元々両プレイヤーに公開されている)。
このモジュールはState全体をそのまま学習に使うと相手の情報が丸見えになってしまう
問題に対応する、State→観測の変換層。
"""

import jax.numpy as jnp

from .state import State

# 非公開(private)フィールド。それ以外(tiles/money/farmer_pos/hands_pos/
# hands_active/hires_today/unlocked_quadrants/market_inventory/town_shop_counts等)
# は元々両プレイヤーに公開されている。
_PRIVATE_FIELDS = ("shed", "seeds", "farmer_inventory", "hands_inventory")


def build_observation(state: State, player) -> State:
    """playerから見た観測を作る。相手(1-player)の非公開フィールドを0で隠す。

    state: [2, ...] 形(step()と同じ、バッチ軸なしの契約)。バッチ実行時は
        jax.vmap(build_observation, in_axes=(0, None))で包む。

    Args:
        state: 現在のState。
        player: 0 or 1。観測を作る対象のプレイヤー。

    Returns:
        stateと同じshapeのState。非公開フィールドは、playerの実際の値は
        そのまま、相手(1-player)の値は0で置き換える。
    """
    opponent = 1 - player
    updates = {field: getattr(state, field).at[opponent].set(0) for field in _PRIVATE_FIELDS}
    return state._replace(**updates)


def build_observations(state: State) -> State:
    """両プレイヤー分の観測を一度に作る。

    self-playなど、両プレイヤーの観測が同時に要る場合の便宜関数
    (build_observation(state, 0)とbuild_observation(state, 1)を
    1つのpytreeにまとめたもの)。

    Args:
        state: [2, ...] 形。

    Returns:
        先頭に「どちらの観測か」を表す軸が付いたState([2, 2, ...])。
        戻り値のインデックス[p]がbuild_observation(state, p)と同じ。
    """
    return State(
        *(
            jnp.stack([a, b])
            for a, b in zip(build_observation(state, 0), build_observation(state, 1), strict=True)
        )
    )
