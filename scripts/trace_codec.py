"""kaggle-environments のトレースJSON(observation/action の生データ)と、
GPU移植版のState/Actionを相互変換する。

validate_against_golden_trace.py から使う(検証ロジック自体はそちらに置く)。
"""

import sys
from pathlib import Path

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kaggriculture.simulator import constants as C
from kaggriculture.simulator.action import Action
from kaggriculture.simulator.state import State


def _dict_to_vector(d, names):
    """{品目名: 個数} の辞書を、namesの並び順に沿った固定長リストに変換する。"""
    return [int(d.get(name, 0)) for name in names]


def _parse_tile(tile, board_size):
    """1マスの生データを、11チャンネル分の値のタプルに変換する。"""
    if tile is None:
        return (C.TILE_EMPTY, -1, 0, False, 0, 0, -1, -1, False, False, 0)
    if tile == "LOCKED":
        return (C.TILE_LOCKED, -1, 0, False, 0, 0, -1, -1, False, False, 0)

    kind = tile["kind"]
    if kind == "WEED":
        return (C.TILE_WEED, -1, 0, False, 0, 0, -1, -1, False, False, 0)
    if kind == "PLANT":
        return (
            C.TILE_PLANT,
            C.CROPS.index(tile["crop"]),
            tile["planted_day"],
            tile["watered_today"],
            tile["consecutive_unwatered"],
            tile["yield_units"],
            tile["max_lifespan_step"],
            tile["fertilized_until_day"],
            False,
            False,
            0,
        )
    # COOP または PASTURE
    tile_kind = C.TILE_COOP if kind == "COOP" else C.TILE_PASTURE
    if "animal" not in tile:
        return (tile_kind, -1, 0, False, 0, 0, -1, -1, False, False, 0)
    return (
        tile_kind,
        C.ANIMALS.index(tile["animal"]),
        tile["placed_day"],
        tile["fed_today"],
        tile["consecutive_unfed"],
        tile["yield_units"],
        -1,
        -1,
        tile["cared_today"],
        tile["fertilizer_available"],
        tile["pending_care_bonus"],
    )


def _parse_farm(farm, private, board_size):
    """1プレイヤー分の farm/private を、Stateの各フィールド名→値の辞書に変換する。"""
    tiles = [
        [_parse_tile(farm["tiles"][y][x], board_size) for x in range(board_size)]
        for y in range(board_size)
    ]
    tile_channels = tuple(
        jnp.array([[cell[i] for cell in row] for row in tiles]) for i in range(11)
    )

    hands_raw = farm["hands"]
    hands_pos = jnp.zeros((C.MAX_HANDS, 2), dtype=jnp.int32)
    hands_active = jnp.zeros((C.MAX_HANDS,), dtype=bool)
    for i, pos in enumerate(hands_raw):
        hands_pos = hands_pos.at[i].set(jnp.array(pos))
        hands_active = hands_active.at[i].set(True)

    inventories = private["inventories"]
    farmer_inventory = jnp.array(
        _dict_to_vector(inventories[0] if inventories else {}, C.SHED_ITEMS)
    )
    hands_inventory = jnp.zeros((C.MAX_HANDS, C.N_SHED_ITEMS), dtype=jnp.int32)
    for i in range(1, len(inventories)):
        hands_inventory = hands_inventory.at[i - 1].set(
            jnp.array(_dict_to_vector(inventories[i], C.SHED_ITEMS))
        )

    unlocked_quadrants = jnp.array([q in farm["unlocked_quadrants"] for q in C.QUADRANTS])

    return {
        "tiles_kind": tile_channels[0],
        "tiles_crop_or_animal": tile_channels[1],
        "tiles_planted_or_placed_day": tile_channels[2],
        "tiles_watered_or_fed_today": tile_channels[3],
        "tiles_consecutive_unwatered_or_unfed": tile_channels[4],
        "tiles_yield_units": tile_channels[5],
        "tiles_max_lifespan_step": tile_channels[6],
        "tiles_fertilized_until_day": tile_channels[7],
        "tiles_cared_today": tile_channels[8],
        "tiles_fertilizer_available": tile_channels[9],
        "tiles_pending_care_bonus": tile_channels[10],
        "farmer_pos": jnp.array(farm["farmer"]),
        "hands_pos": hands_pos,
        "hands_active": hands_active,
        "hires_today": jnp.array(farm["hires_today"]),
        "money": jnp.array(float(farm["money"])),
        "unlocked_quadrants": unlocked_quadrants,
        "shed": jnp.array(_dict_to_vector(private["shed"], C.SHED_ITEMS)),
        "seeds": jnp.array(_dict_to_vector(private["seeds"], C.CROPS)),
        "farmer_inventory": farmer_inventory,
        "hands_inventory": hands_inventory,
    }


def _stack_player_fields(p0, p1, keys):
    """プレイヤー0/1分の {フィールド名: 値} 辞書2つを、[2, ...] 形にまとめる。"""
    return {k: jnp.stack([p0[k], p1[k]]) for k in keys}


def build_state(obs0_player0, obs0_player1, board_size):
    """両プレイヤー分のobservationから、1ターン分のState([2, ...]形)を組み立てる。"""
    p0 = _parse_farm(obs0_player0["farms"][0], obs0_player0["private"], board_size)
    p1 = _parse_farm(obs0_player1["farms"][1], obs0_player1["private"], board_size)
    keys = list(p0.keys())
    stacked = _stack_player_fields(p0, p1, keys)

    market_inventory = jnp.array(_dict_to_vector(obs0_player0["market"]["inventory"], C.PRODUCTS))
    town_shop_counts = jnp.zeros(C.N_SHOPS, dtype=jnp.int32)
    for shop in obs0_player0["town"]["unlocked_shops"]:
        town_shop_counts = town_shop_counts.at[C.SHOPS.index(shop)].add(1)

    return State(
        tiles_kind=stacked["tiles_kind"],
        tiles_crop_or_animal=stacked["tiles_crop_or_animal"],
        tiles_planted_or_placed_day=stacked["tiles_planted_or_placed_day"],
        tiles_watered_or_fed_today=stacked["tiles_watered_or_fed_today"],
        tiles_consecutive_unwatered_or_unfed=stacked["tiles_consecutive_unwatered_or_unfed"],
        tiles_yield_units=stacked["tiles_yield_units"],
        tiles_max_lifespan_step=stacked["tiles_max_lifespan_step"],
        tiles_fertilized_until_day=stacked["tiles_fertilized_until_day"],
        tiles_cared_today=stacked["tiles_cared_today"],
        tiles_fertilizer_available=stacked["tiles_fertilizer_available"],
        tiles_pending_care_bonus=stacked["tiles_pending_care_bonus"],
        farmer_pos=stacked["farmer_pos"],
        hands_pos=stacked["hands_pos"],
        hands_active=stacked["hands_active"],
        hires_today=stacked["hires_today"],
        money=stacked["money"],
        unlocked_quadrants=stacked["unlocked_quadrants"],
        shed=stacked["shed"],
        seeds=stacked["seeds"],
        farmer_inventory=stacked["farmer_inventory"],
        hands_inventory=stacked["hands_inventory"],
        market_inventory=market_inventory,
        town_shop_counts=town_shop_counts,
        step=jnp.array(obs0_player0["step"]),
        rng_key=jnp.zeros(
            (2,), dtype=jnp.uint32
        ),  # このスクリプトでは日末処理が無い範囲だけ検証するため未使用
    )


def _parse_unit_action(action_list):
    """[op, ...args] を (op_idx, arg_idx, n) に変換する。"""
    if not action_list:
        return C.FARMER_OP_PASS, 0, 0
    op = C.FARMER_OP_NAMES.index(action_list[0])
    arg_idx, n = 0, 0
    if op == C.FARMER_OP_PLANT and len(action_list) >= 2:
        arg_idx = C.CROPS.index(action_list[1])
    elif op in (C.FARMER_OP_PICKUP, C.FARMER_OP_PLACE) and len(action_list) >= 2:
        arg_idx = C.SHED_ITEMS.index(action_list[1])
        n = int(action_list[2]) if len(action_list) >= 3 else 1
    return op, arg_idx, n


_MARKET_ITEM_TABLE = {
    C.MARKET_OP_BUY_SEED: C.CROPS,
    C.MARKET_OP_BUY_PRODUCT: C.PRODUCTS,
    C.MARKET_OP_BUY_ANIMAL: C.ANIMALS,
    C.MARKET_OP_SELL: C.PRODUCTS,
}


def _parse_market_orders(order_list):
    """market注文リストを [MAX_MARKET_ORDERS] の (op, item, n) 配列に変換する。"""
    ops = [-1] * C.MAX_MARKET_ORDERS
    items = [0] * C.MAX_MARKET_ORDERS
    ns = [0] * C.MAX_MARKET_ORDERS
    for i, order in enumerate(order_list[: C.MAX_MARKET_ORDERS]):
        op = C.MARKET_OP_NAMES.index(order[0])
        ops[i] = op
        if op in _MARKET_ITEM_TABLE:
            items[i] = _MARKET_ITEM_TABLE[op].index(order[1])
            ns[i] = int(order[2])
    return ops, items, ns


def build_action(action0, action1):
    """両プレイヤー分の行動(トレースの生データ)から、1ターン分のActionを組み立てる。"""
    farmer0 = _parse_unit_action(action0.get("farmer", ["PASS"]))
    farmer1 = _parse_unit_action(action1.get("farmer", ["PASS"]))

    hands_op = jnp.full((2, C.MAX_HANDS), C.FARMER_OP_PASS, dtype=jnp.int32)
    hands_arg = jnp.zeros((2, C.MAX_HANDS), dtype=jnp.int32)
    hands_n = jnp.zeros((2, C.MAX_HANDS), dtype=jnp.int32)
    for p, action in enumerate([action0, action1]):
        for i, hand_action in enumerate(action.get("hands", [])):
            op, arg, n = _parse_unit_action(hand_action)
            hands_op = hands_op.at[p, i].set(op)
            hands_arg = hands_arg.at[p, i].set(arg)
            hands_n = hands_n.at[p, i].set(n)

    mop0, mitem0, mn0 = _parse_market_orders(action0.get("market", []))
    mop1, mitem1, mn1 = _parse_market_orders(action1.get("market", []))

    return Action(
        farmer_op=jnp.array([farmer0[0], farmer1[0]], dtype=jnp.int32),
        farmer_arg_idx=jnp.array([farmer0[1], farmer1[1]], dtype=jnp.int32),
        farmer_n=jnp.array([farmer0[2], farmer1[2]], dtype=jnp.int32),
        hands_op=hands_op,
        hands_arg_idx=hands_arg,
        hands_n=hands_n,
        market_op=jnp.array([mop0, mop1], dtype=jnp.int32),
        market_arg_idx=jnp.array([mitem0, mitem1], dtype=jnp.int32),
        market_n=jnp.array([mn0, mn1], dtype=jnp.int32),
    )
