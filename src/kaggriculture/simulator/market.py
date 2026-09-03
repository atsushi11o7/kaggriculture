"""市場価格の計算(元コードの market_price / _shape)。"""

import jax.numpy as jnp

from . import constants as C
from . import game_params as P

_BASE = jnp.array(P.MARKET_BASE_PRICE, dtype=jnp.float32)
_T = jnp.array(P.MARKET_T, dtype=jnp.float32)
_BELOW_FUNC = jnp.array(P.MARKET_BELOW_FUNC, dtype=jnp.int32)
_BELOW_TARGET = jnp.array(P.MARKET_BELOW_TARGET, dtype=jnp.float32)
_ABOVE_FUNC = jnp.array(P.MARKET_ABOVE_FUNC, dtype=jnp.int32)
_ABOVE_TARGET = jnp.array(P.MARKET_ABOVE_TARGET, dtype=jnp.float32)


def _shape(func_code, x, t):
    """価格カーブの形状関数を評価する。

    Args:
        func_code: constants.FUNC_*。xと同じ形にブロードキャスト可能。
        x: 非負にクリップされる入力値。
        t: hinge関数でのみ使う基準値。

    Returns:
        f(x)。
    """
    x = jnp.maximum(0.0, x)
    t_safe = jnp.where(t > 0, t, 1.0)
    u = x / t_safe
    hinge = jnp.where(t > 0, u + P.HINGE_GAIN * jnp.maximum(0.0, u - 1.0) ** 2, x)

    return jnp.select(
        [
            func_code == C.FUNC_LINEAR,
            func_code == C.FUNC_SQ,
            func_code == C.FUNC_SQRT,
            func_code == C.FUNC_LOG,
            func_code == C.FUNC_LOG10,
            func_code == C.FUNC_HINGE,
        ],
        [x, x * x, jnp.sqrt(x), jnp.log1p(x), jnp.log10(1.0 + x), hinge],
        default=x,
    )


def market_price(inventory):
    """在庫量から品目ごとの価格を計算する(元コードの market_price + _refresh_prices)。

    在庫が基準値I0を下回れば below_func/below_target、上回れば above_func/above_target
    を使う。PRICE_FLOORで下限、整数に丸める。

    Args:
        inventory: [N_PRODUCTS] 市場在庫。

    Returns:
        [N_PRODUCTS] 価格。
    """
    inventory = inventory.astype(jnp.float32)
    below = inventory < P.MARKET_I0

    func = jnp.where(below, _BELOW_FUNC, _ABOVE_FUNC)
    target = jnp.where(below, _BELOW_TARGET, _ABOVE_TARGET)
    sign = jnp.where(below, 1.0, -1.0)
    x = jnp.where(below, P.MARKET_I0 - inventory, inventory - P.MARKET_I0)

    amp = target * _BASE / _shape(func, _T, _T)
    price = _BASE + sign * amp * _shape(func, x, _T)

    return jnp.maximum(P.PRICE_FLOOR, jnp.round(price)).astype(jnp.int32)


def market_price_one(item, inventory_value):
    """在庫量から1品目だけの価格を計算する。

    market_price()は全N_PRODUCTS品目分を計算するので、1品目の価格だけ欲しい場面
    (市場のロックステップ処理など、1ティックにつき1品目しか使わない場面)では
    無駄が大きい。先にitem番目のパラメータだけ取り出してからスカラーで計算する。

    Args:
        item: 品目(constants.PRODUCTS)のインデックス。
        inventory_value: その品目の市場在庫(スカラー)。

    Returns:
        価格(スカラー)。
    """
    inventory_value = inventory_value.astype(jnp.float32)
    below = inventory_value < P.MARKET_I0

    base = _BASE[item]
    t = _T[item]
    func = jnp.where(below, _BELOW_FUNC[item], _ABOVE_FUNC[item])
    target = jnp.where(below, _BELOW_TARGET[item], _ABOVE_TARGET[item])
    sign = jnp.where(below, 1.0, -1.0)
    x = jnp.where(below, P.MARKET_I0 - inventory_value, inventory_value - P.MARKET_I0)

    amp = target * base / _shape(func, t, t)
    price = base + sign * amp * _shape(func, x, t)

    return jnp.maximum(P.PRICE_FLOOR, jnp.round(price)).astype(jnp.int32)
