"""SELL/BUY_PRODUCT注文のロックステップ処理(元コードの_process_marketのwhileループ)。

SELL/BUY_PRODUCTは価格が市場在庫に応じて売買のたびに動くため、他の市場注文
(BUY_SEED/BUY_ANIMAL)のような閉じた式では計算できず、1単位ずつ両プレイヤー
同時に処理する必要がある。

ループ回数の上限はshed_capacity(SELLは手持ち、BUY_PRODUCTは納屋の空きが
尽きれば止まるため、これを超えて有効な取引は続かない)。jax.lax.while_loopで
両者とも非アクティブになった時点で打ち切る(実測: batch=1〜2000で1.4〜1.75倍
高速化。vmapでバッチ化すると1局でも長く続く局があれば全体がその反復数を
待つため、削減幅は市場注文の分布に依存する)。while_loopはlax.scanと違い
逆伝播できない(このシミュレータは方策側にしか勾配が要らないRL用途を想定して
おり、通常は問題にならない)。
"""

import jax
import jax.numpy as jnp

from . import constants as C
from .market import market_price_one

_WHEAT_IDX = C.PRODUCTS.index("WHEAT")
_FERTILIZER_IDX = C.PRODUCTS.index("FERTILIZER")


def _is_active(op, item, n):
    """この注文がロックステップに参加するか(元コードのquoted判定)。"""
    is_sell = op == C.MARKET_OP_SELL
    is_buy_product = (op == C.MARKET_OP_BUY_PRODUCT) & (
        (item == _WHEAT_IDX) | (item == _FERTILIZER_IDX)
    )
    return (is_sell | is_buy_product) & (n > 0)


def _quote_and_commit(op, item, money, shed, active, market_inventory, shed_capacity):
    """1ユニット分の見積もり・コミットを1プレイヤー分行う。

    Returns:
        (new_money, new_shed, market_inventory_delta, still_active)。
    """
    is_sell = op == C.MARKET_OP_SELL

    # 全品目分ではなく、この品目1つだけを計算する(市場処理は1ティックにつき最大
    # shed_capacity回呼ばれるので、9品目分丸ごと計算するのは無駄が大きい)。
    sell_price = market_price_one(item, market_inventory[item])
    buy_price = market_price_one(item, market_inventory[item] - 1)  # 購入後在庫で見積もる
    price = jnp.where(is_sell, sell_price, buy_price)

    can_sell = shed[item] > 0
    can_buy = (money >= price) & (jnp.sum(shed) < shed_capacity)
    can_commit = active & jnp.where(is_sell, can_sell, can_buy)

    # itemはPRODUCTSのインデックスで、SHED_ITEMS(=PRODUCTS+ANIMALS)の先頭9個と
    # 同じ並びなので、shed用とmarket_inventory用は長さの違うone-hotが要る。
    shed_one_hot = jax.nn.one_hot(item, C.N_SHED_ITEMS, dtype=jnp.int32)
    product_one_hot = jax.nn.one_hot(item, C.N_PRODUCTS, dtype=jnp.int32)

    money_delta = jnp.where(is_sell, price, -price)
    new_money = money + jnp.where(can_commit, money_delta, 0.0)

    shed_delta = jnp.where(is_sell, -shed_one_hot, shed_one_hot)
    new_shed = shed + jnp.where(can_commit, shed_delta, 0)

    # SELLは市場在庫を+1するが、$1の床値では増やさない(在庫を増やさないことで
    # 床値が以降の買いにも反応し続ける)。BUYは常に-1。
    inv_delta_sell = jnp.where(can_commit & (price > 1), product_one_hot, 0)
    inv_delta_buy = jnp.where(can_commit, -product_one_hot, 0)
    inv_delta = jnp.where(is_sell, inv_delta_sell, inv_delta_buy)

    return new_money, new_shed, inv_delta, can_commit


def market_lockstep(
    op_a,
    item_a,
    n_a,
    money_a,
    shed_a,
    op_b,
    item_b,
    n_b,
    money_b,
    shed_b,
    market_inventory,
    shed_capacity,
):
    """SELL/BUY_PRODUCT注文を両プレイヤー同時に1単位ずつ処理する。

    Args:
        op_a, item_a, n_a, money_a, shed_a: プレイヤーAの注文と状態。
        op_b, item_b, n_b, money_b, shed_b: プレイヤーBの注文と状態。
        market_inventory: [N_PRODUCTS] 市場在庫。
        shed_capacity: 納屋の容量(ループ回数の上限としても使う)。

    Returns:
        (new_money_a, new_shed_a, new_money_b, new_shed_b, new_market_inventory)。
    """
    active_a0 = _is_active(op_a, item_a, n_a)
    active_b0 = _is_active(op_b, item_b, n_b)

    def cond(carry):
        (_, _, _, _, active_a, _, _, _, active_b, i) = carry
        return (active_a | active_b) & (i < shed_capacity)

    def tick(carry):
        (
            market_inv,
            money_a,
            shed_a,
            remaining_a,
            active_a,
            money_b,
            shed_b,
            remaining_b,
            active_b,
            i,
        ) = carry

        money_a, shed_a, delta_a, committed_a = _quote_and_commit(
            op_a, item_a, money_a, shed_a, active_a, market_inv, shed_capacity
        )
        money_b, shed_b, delta_b, committed_b = _quote_and_commit(
            op_b, item_b, money_b, shed_b, active_b, market_inv, shed_capacity
        )
        market_inv = market_inv + delta_a + delta_b

        remaining_a = jnp.where(committed_a, remaining_a - 1, remaining_a)
        remaining_b = jnp.where(committed_b, remaining_b - 1, remaining_b)
        active_a = committed_a & (remaining_a > 0)
        active_b = committed_b & (remaining_b > 0)

        return (
            market_inv,
            money_a,
            shed_a,
            remaining_a,
            active_a,
            money_b,
            shed_b,
            remaining_b,
            active_b,
            i + 1,
        )

    init_carry = (
        market_inventory,
        money_a,
        shed_a,
        n_a,
        active_a0,
        money_b,
        shed_b,
        n_b,
        active_b0,
        0,
    )
    final_carry = jax.lax.while_loop(cond, tick, init_carry)
    market_inv, money_a, shed_a, _, _, money_b, shed_b, _, _, _ = final_carry

    return money_a, shed_a, money_b, shed_b, market_inv
