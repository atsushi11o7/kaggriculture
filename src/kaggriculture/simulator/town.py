"""街(店舗・タウンセンター)による市場消費(元コードの _town_consume)。"""

import jax.numpy as jnp

from . import game_params as P

_SHOP_DEMAND = jnp.array(P.SHOP_DEMAND, dtype=jnp.int32)  # [N_SHOPS, N_PRODUCTS]
_TOWN_CENTER_PRODUCTS = jnp.array(P.TOWN_CENTER_PRODUCTS, dtype=jnp.int32)  # [N_PRODUCTS]


def apply_town_consumption(
    market_inventory, town_shop_counts, step, shop_interval, center_interval
):
    """街による市場在庫の消費を1ターン分適用する。

    店はshop_intervalターンごとに、出現している各店が扱う品目を消費する
    (SHOP_DEMANDに単品目店の2倍ルールを含む)。タウンセンターはcenter_interval
    ターンごとに肥料以外の全品目を1ずつ消費する。乱数は使わない(店の抽選は
    別処理)。

    Args:
        market_inventory: [N_PRODUCTS] 市場在庫。
        town_shop_counts: [N_SHOPS] 店の種類ごとの出現数。
        step: 現在の通しターン数。
        shop_interval: 店が消費するターン間隔。
        center_interval: タウンセンターが消費するターン間隔。

    Returns:
        new_market_inventory。
    """
    shop_tick = (step % shop_interval) == 0
    center_tick = (step % center_interval) == 0

    shop_consumption = town_shop_counts @ _SHOP_DEMAND
    consumption = jnp.where(shop_tick, shop_consumption, 0) + jnp.where(
        center_tick, _TOWN_CENTER_PRODUCTS, 0
    )
    return market_inventory - consumption
