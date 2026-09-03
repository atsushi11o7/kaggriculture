"""1ターン分の行動(Action)。State同様、[2, ...]の先頭軸はプレイヤー。"""

from typing import NamedTuple

import jax.numpy as jnp


class Action(NamedTuple):
    # farmerの行動。arg_idxの意味はopに依存する(PLANTなら作物、PICKUP/PLACEなら品目)。
    farmer_op: jnp.ndarray  # [2]
    farmer_arg_idx: jnp.ndarray  # [2]
    farmer_n: jnp.ndarray  # [2]

    # 各handの行動。[2, MAX_HANDS]。hands_active=Falseのスロットは無視される。
    hands_op: jnp.ndarray
    hands_arg_idx: jnp.ndarray
    hands_n: jnp.ndarray

    # 市場注文キュー。[2, MAX_MARKET_ORDERS]。スロット0から順に処理される。
    market_op: jnp.ndarray
    market_arg_idx: jnp.ndarray
    market_n: jnp.ndarray
