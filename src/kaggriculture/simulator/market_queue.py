"""市場注文キューを両プレイヤー分、順番に処理する(元コードの_process_marketの外側ループ)。

1キュースロットにつき、注文の種類ごとに以下の順で適用する:
- HIRE / BUY_LAND: 即時1回(atomic)
- BUY_SEED / BUY_ANIMAL: 閉じた式(market_orders.py)
- SELL / BUY_PRODUCT: ロックステップ(market_lockstep.py)

これらは1プレイヤーにつき同時に1種類の注文しか無く(opは1つ)、かつ
BUY_SEED/BUY_ANIMALは市場在庫やお互いの資源と無関係なので、この順で
呼んでも結果は変わらない(unit_actions.apply_unit_actionと同じ「該当しない
関数はno-op」の合成パターン)。スロット間はプレイヤーの所持金・納屋などが
引き継がれるため、jax.lax.scanで逐次処理する。
"""

import jax

from .market_lockstep import market_lockstep
from .market_orders import apply_buy_animal, apply_buy_land, apply_buy_seed, apply_hire


def _process_slot(
    op_a,
    item_a,
    n_a,
    op_b,
    item_b,
    n_b,
    money_a,
    money_b,
    hires_today_a,
    hires_today_b,
    hands_active_a,
    hands_active_b,
    hands_pos_a,
    hands_pos_b,
    farmer_pos_a,
    farmer_pos_b,
    unlocked_quadrants_a,
    unlocked_quadrants_b,
    tile_kind_a,
    tile_kind_b,
    seeds_a,
    seeds_b,
    shed_a,
    shed_b,
    market_inventory,
    board_size,
    hire_mult,
    shed_capacity,
):
    """1キュースロット分、両プレイヤーの注文を適用する。"""
    money_a, hires_today_a, hands_active_a, hands_pos_a = apply_hire(
        op_a,
        money_a,
        hires_today_a,
        hands_active_a,
        hands_pos_a,
        farmer_pos_a,
        board_size,
        hire_mult,
    )
    money_b, hires_today_b, hands_active_b, hands_pos_b = apply_hire(
        op_b,
        money_b,
        hires_today_b,
        hands_active_b,
        hands_pos_b,
        farmer_pos_b,
        board_size,
        hire_mult,
    )

    money_a, unlocked_quadrants_a, tile_kind_a = apply_buy_land(
        op_a, money_a, unlocked_quadrants_a, tile_kind_a, board_size
    )
    money_b, unlocked_quadrants_b, tile_kind_b = apply_buy_land(
        op_b, money_b, unlocked_quadrants_b, tile_kind_b, board_size
    )

    money_a, seeds_a = apply_buy_seed(op_a, item_a, n_a, money_a, seeds_a)
    money_b, seeds_b = apply_buy_seed(op_b, item_b, n_b, money_b, seeds_b)

    money_a, shed_a = apply_buy_animal(op_a, item_a, n_a, money_a, shed_a, shed_capacity)
    money_b, shed_b = apply_buy_animal(op_b, item_b, n_b, money_b, shed_b, shed_capacity)

    money_a, shed_a, money_b, shed_b, market_inventory = market_lockstep(
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
    )

    return (
        money_a,
        money_b,
        hires_today_a,
        hires_today_b,
        hands_active_a,
        hands_active_b,
        hands_pos_a,
        hands_pos_b,
        unlocked_quadrants_a,
        unlocked_quadrants_b,
        tile_kind_a,
        tile_kind_b,
        seeds_a,
        seeds_b,
        shed_a,
        shed_b,
        market_inventory,
    )


def process_market_queue(
    queue_op_a,
    queue_item_a,
    queue_n_a,
    queue_op_b,
    queue_item_b,
    queue_n_b,
    money_a,
    money_b,
    hires_today_a,
    hires_today_b,
    hands_active_a,
    hands_active_b,
    hands_pos_a,
    hands_pos_b,
    farmer_pos_a,
    farmer_pos_b,
    unlocked_quadrants_a,
    unlocked_quadrants_b,
    tile_kind_a,
    tile_kind_b,
    seeds_a,
    seeds_b,
    shed_a,
    shed_b,
    market_inventory,
    board_size,
    hire_mult,
    shed_capacity,
):
    """両プレイヤーの市場注文キューを、スロット0から順に処理する。

    Args:
        queue_op_a, queue_item_a, queue_n_a: [MAX_MARKET_ORDERS] プレイヤーAの注文キュー。
        queue_op_b, queue_item_b, queue_n_b: [MAX_MARKET_ORDERS] プレイヤーBの注文キュー。
        money_a, money_b: 所持金。
        hires_today_a, hires_today_b: 今日すでに雇った人数。
        hands_active_a, hands_active_b: [MAX_HANDS] bool。
        hands_pos_a, hands_pos_b: [MAX_HANDS, 2]。
        farmer_pos_a, farmer_pos_b: [2]。
        unlocked_quadrants_a, unlocked_quadrants_b: [N_QUADRANTS] bool。
        tile_kind_a, tile_kind_b: [board, board] constants.TILE_*。
        seeds_a, seeds_b: [N_CROPS]。
        shed_a, shed_b: [N_SHED_ITEMS]。
        market_inventory: [N_PRODUCTS]。
        board_size, hire_mult, shed_capacity: 共通パラメータ。

    Returns:
        (money_a, money_b, hires_today_a, hires_today_b, hands_active_a, hands_active_b,
        hands_pos_a, hands_pos_b, unlocked_quadrants_a, unlocked_quadrants_b,
        tile_kind_a, tile_kind_b, seeds_a, seeds_b, shed_a, shed_b, market_inventory)。
    """

    def step(carry, slot):
        op_a, item_a, n_a, op_b, item_b, n_b = slot
        (
            money_a,
            money_b,
            hires_today_a,
            hires_today_b,
            hands_active_a,
            hands_active_b,
            hands_pos_a,
            hands_pos_b,
            unlocked_quadrants_a,
            unlocked_quadrants_b,
            tile_kind_a,
            tile_kind_b,
            seeds_a,
            seeds_b,
            shed_a,
            shed_b,
            market_inventory,
        ) = carry

        new_carry = _process_slot(
            op_a,
            item_a,
            n_a,
            op_b,
            item_b,
            n_b,
            money_a,
            money_b,
            hires_today_a,
            hires_today_b,
            hands_active_a,
            hands_active_b,
            hands_pos_a,
            hands_pos_b,
            farmer_pos_a,
            farmer_pos_b,
            unlocked_quadrants_a,
            unlocked_quadrants_b,
            tile_kind_a,
            tile_kind_b,
            seeds_a,
            seeds_b,
            shed_a,
            shed_b,
            market_inventory,
            board_size,
            hire_mult,
            shed_capacity,
        )
        return new_carry, None

    init_carry = (
        money_a,
        money_b,
        hires_today_a,
        hires_today_b,
        hands_active_a,
        hands_active_b,
        hands_pos_a,
        hands_pos_b,
        unlocked_quadrants_a,
        unlocked_quadrants_b,
        tile_kind_a,
        tile_kind_b,
        seeds_a,
        seeds_b,
        shed_a,
        shed_b,
        market_inventory,
    )
    xs = (queue_op_a, queue_item_a, queue_n_a, queue_op_b, queue_item_b, queue_n_b)
    final_carry, _ = jax.lax.scan(step, init_carry, xs)
    return final_carry
