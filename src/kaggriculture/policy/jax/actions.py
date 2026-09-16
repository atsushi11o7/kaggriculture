"""JAX方策用の固定候補テーブルと合法手mask。"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.jax import features as F
from kaggriculture.rules import constants as C
from kaggriculture.rules import game_params as P
from kaggriculture.simulator import market, market_orders
from kaggriculture.simulator.market_lockstep import market_lockstep
from kaggriculture.simulator.state import State
from kaggriculture.simulator.unit_actions import apply_unit_action

MARKET_WAIT = C.N_MARKET_OPS
MARKET_STOP = C.N_MARKET_OPS + 1


class CandidateTable(NamedTuple):
    """固定候補の環境用表現と埋め込み用疎特徴。"""

    op: jnp.ndarray
    arg: jnp.ndarray
    index: jnp.ndarray
    value: jnp.ndarray


def _candidate_features(ops, args, market_action: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
    n = len(ops)
    index = jnp.zeros((n, F.MAX_CANDIDATE_FEATURES), dtype=jnp.int32)
    value = jnp.zeros((n, F.MAX_CANDIDATE_FEATURES), dtype=jnp.float32)
    for row, (op, arg) in enumerate(zip(ops, args, strict=True)):
        if market_action and op == MARKET_WAIT:
            index = index.at[row, 0].set(V.ACTION_MARKET_WAIT.start)
            value = value.at[row, 0].set(1.0)
            continue
        if market_action and op == MARKET_STOP:
            index = index.at[row, 0].set(V.ACTION_MARKET_STOP.start)
            value = value.at[row, 0].set(1.0)
            continue
        table = V.ACTION_MARKET_OP if market_action else V.ACTION_FARMER_OP
        index = index.at[row, 0].set(table.start + op)
        value = value.at[row, 0].set(1.0)
        if arg >= 0:
            if not market_action and op == C.FARMER_OP_PLANT:
                entity = arg
            elif market_action and op == C.MARKET_OP_BUY_SEED:
                entity = arg
            elif market_action and op == C.MARKET_OP_BUY_ANIMAL:
                entity = C.N_PRODUCTS + arg
            else:
                entity = arg
            index = index.at[row, 1].set(V.ENTITY_ITEM.start + entity)
            value = value.at[row, 1].set(1.0)
    return index, value


def _build_unit_candidates() -> CandidateTable:
    item_ops = {C.FARMER_OP_PICKUP, C.FARMER_OP_PLANT, C.FARMER_OP_PLACE}
    ops = [op for op in range(C.N_FARMER_OPS) if op not in item_ops]
    args = [-1] * len(ops)
    ops.extend([C.FARMER_OP_PLANT] * C.N_CROPS)
    args.extend(range(C.N_CROPS))
    ops.extend([C.FARMER_OP_PLACE] * C.N_SHED_ITEMS)
    args.extend(range(C.N_SHED_ITEMS))
    ops.extend([C.FARMER_OP_PICKUP] * C.N_SHED_ITEMS)
    args.extend(range(C.N_SHED_ITEMS))
    index, value = _candidate_features(ops, args, market_action=False)
    return CandidateTable(jnp.asarray(ops), jnp.asarray(args), index, value)


def _build_market_candidates() -> CandidateTable:
    ops = [C.MARKET_OP_HIRE, C.MARKET_OP_BUY_LAND]
    args = [-1, -1]
    ops.extend([C.MARKET_OP_BUY_SEED] * C.N_CROPS)
    args.extend(range(C.N_CROPS))
    ops.extend([C.MARKET_OP_BUY_ANIMAL] * C.N_ANIMALS)
    args.extend(range(C.N_ANIMALS))
    for item in (C.PRODUCTS.index("WHEAT"), C.PRODUCTS.index("FERTILIZER")):
        ops.append(C.MARKET_OP_BUY_PRODUCT)
        args.append(item)
    ops.extend([C.MARKET_OP_SELL] * C.N_PRODUCTS)
    args.extend(range(C.N_PRODUCTS))
    ops.extend([MARKET_WAIT, MARKET_STOP])
    args.extend([-1, -1])
    index, value = _candidate_features(ops, args, market_action=True)
    return CandidateTable(jnp.asarray(ops), jnp.asarray(args), index, value)


UNIT_CANDIDATES = _build_unit_candidates()
MARKET_CANDIDATES = _build_market_candidates()


def quantity_candidates() -> tuple[jnp.ndarray, jnp.ndarray]:
    """数量1..MAX_ACTION_QUANTITYの固定候補特徴を返す。"""
    quantity = jnp.arange(1, V.MAX_ACTION_QUANTITY + 1)
    index = jnp.zeros((V.MAX_ACTION_QUANTITY, F.MAX_CANDIDATE_FEATURES), dtype=jnp.int32)
    value = jnp.zeros((V.MAX_ACTION_QUANTITY, F.MAX_CANDIDATE_FEATURES), dtype=jnp.float32)
    index = index.at[:, 0].set(V.ACTION_QUANTITY_VALUE.start + quantity - 1)
    index = index.at[:, 1].set(V.ACTION_QUANTITY_CONTINUOUS.start)
    value = value.at[:, 0].set(1.0)
    value = value.at[:, 1].set(
        jnp.log1p(quantity) / jnp.log1p(jnp.asarray(V.MAX_ACTION_QUANTITY, jnp.float32))
    )
    return index, value


QUANTITY_INDEX, QUANTITY_VALUE = quantity_candidates()
_fib_values = [1, 1]
for _ in range(C.MAX_HANDS - 1):
    _fib_values.append(_fib_values[-1] + _fib_values[-2])
_FIBONACCI = jnp.asarray(_fib_values, dtype=jnp.int32)
_LAND_PRICES = jnp.asarray(P.LAND_PRICES, dtype=jnp.float32)
_SEED_COSTS = jnp.asarray(P.CROP_SEED_COST, dtype=jnp.float32)
_ANIMAL_COSTS = jnp.asarray(P.ANIMAL_COST, dtype=jnp.float32)
_FIRST_YIELD_DAY = jnp.asarray(P.CROP_FIRST_YIELD_DAY, dtype=jnp.int32)
_MOVE_DELTA = jnp.asarray(C.FARMER_OP_MOVE_DELTA, dtype=jnp.int32)


def unit_fields(state: State, player: jnp.ndarray, unit: jnp.ndarray):
    is_farmer = unit == 0
    hand = jnp.maximum(unit - 1, 0)
    pos = jnp.where(is_farmer, state.farmer_pos[player], state.hands_pos[player, hand])
    inventory = jnp.where(
        is_farmer,
        state.farmer_inventory[player],
        state.hands_inventory[player, hand],
    )
    return pos, inventory


def _shed_adjacent(pos: jnp.ndarray, board_size: int) -> jnp.ndarray:
    half = board_size // 2
    return ((pos[0] == half - 1) | (pos[0] == half)) & ((pos[1] == half - 1) | (pos[1] == half))


def _animal_placement(kind, tile_item, item: jnp.ndarray) -> jnp.ndarray:
    animal = item - C.N_PRODUCTS
    is_animal = (animal >= 0) & (animal < C.N_ANIMALS)
    correct_structure = jnp.where(
        animal == C.ANIMALS.index("GOOSE"),
        kind == C.TILE_COOP,
        kind == C.TILE_PASTURE,
    )
    return is_animal & correct_structure & (tile_item < 0)


def legal_unit_mask(
    state: State,
    player: jnp.ndarray,
    unit: jnp.ndarray,
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
) -> jnp.ndarray:
    """1ユニットの固定候補に対する合法maskを返す。"""
    pos, inventory = unit_fields(state, player, unit)
    x, y = pos
    kind = state.tiles_kind[player, y, x]
    tile_item = state.tiles_crop_or_animal[player, y, x]
    placed_day = state.tiles_planted_or_placed_day[player, y, x]
    cared = state.tiles_watered_or_fed_today[player, y, x]
    yields = state.tiles_yield_units[player, y, x]
    animal_cared = state.tiles_cared_today[player, y, x]
    fertilizer_available = state.tiles_fertilizer_available[player, y, x]
    day = state.step // turns_per_day
    adjacent = _shed_adjacent(pos, state.tiles_kind.shape[-1])
    op, arg = UNIT_CANDIDATES.op, UNIT_CANDIDATES.arg

    candidate_pos = pos + _MOVE_DELTA[jnp.clip(op, 0, C.N_FARMER_OPS - 1)]
    in_bounds = jnp.all(
        (candidate_pos >= 0) & (candidate_pos < state.tiles_kind.shape[-1]), axis=-1
    )
    is_plant = kind == C.TILE_PLANT
    is_structure = (kind == C.TILE_COOP) | (kind == C.TILE_PASTURE)
    has_animal = is_structure & (tile_item >= 0)
    empty = kind == C.TILE_EMPTY
    animal_place = _animal_placement(kind, tile_item, arg)
    item_arg = jnp.clip(arg, 0, C.N_SHED_ITEMS - 1)
    crop_arg = jnp.clip(arg, 0, C.N_CROPS - 1)
    mature = (day - placed_day) >= _FIRST_YIELD_DAY[jnp.clip(tile_item, 0, C.N_CROPS - 1)]

    legal = op == C.FARMER_OP_PASS
    move = (op >= C.FARMER_OP_NORTH) & (op <= C.FARMER_OP_WEST)
    legal |= move & in_bounds
    # 納屋の共有在庫と空き容量はExecutorがunit順に解決する。
    legal |= (op == C.FARMER_OP_PICKUP) & adjacent
    legal |= (op == C.FARMER_OP_PLANT) & empty & (state.seeds[player, crop_arg] > 0)
    legal |= (op == C.FARMER_OP_WATER) & is_plant & ~cared
    legal |= (op == C.FARMER_OP_HARVEST) & (
        (is_plant & mature & (yields > 0)) | (has_animal & (yields > 0))
    )
    fertilizer_idx = C.SHED_ITEMS.index("FERTILIZER")
    legal |= (op == C.FARMER_OP_FERTILIZE) & is_plant & (inventory[fertilizer_idx] > 0)
    legal |= (op == C.FARMER_OP_BUILD_COOP) & empty
    legal |= (op == C.FARMER_OP_BUILD_PASTURE) & empty
    legal |= (op == C.FARMER_OP_DIG) & (
        is_plant | (kind == C.TILE_WEED) | (is_structure & ~has_animal)
    )
    shed_drop = adjacent & (inventory[item_arg] > 0) & ~animal_place
    legal |= (op == C.FARMER_OP_PLACE) & ((animal_place & (inventory[item_arg] > 0)) | shed_drop)
    wheat_idx = C.SHED_ITEMS.index("WHEAT")
    legal |= (op == C.FARMER_OP_FEED) & has_animal & ~cared & (inventory[wheat_idx] > 0)
    legal |= (op == C.FARMER_OP_COLLECT_FERTILIZER) & has_animal & fertilizer_available
    legal |= (op == C.FARMER_OP_CARE) & has_animal & ~animal_cared
    legal |= (op == C.FARMER_OP_DROP) & adjacent & jnp.any(inventory > 0)
    return legal


def legal_market_mask(
    state: State,
    player: jnp.ndarray,
    *,
    hire_mult: float = 1.0,
    shed_capacity: int = 100,
) -> jnp.ndarray:
    """市場固定候補に対する合法maskを返す。"""
    op, arg = MARKET_CANDIDATES.op, MARKET_CANDIDATES.arg
    money = state.money[player]
    shed = state.shed[player]
    room = jnp.sum(shed) < shed_capacity
    active_hands = jnp.sum(state.hands_active[player])
    hires = jnp.clip(state.hires_today[player], 0, C.MAX_HANDS)
    legal = (op == MARKET_WAIT) | (op == MARKET_STOP)
    legal |= (
        (op == C.MARKET_OP_HIRE)
        & (active_hands < C.MAX_HANDS)
        & (money >= _FIBONACCI[hires] * hire_mult)
    )
    n_extra = jnp.sum(state.unlocked_quadrants[player]) - 1
    land_index = jnp.clip(n_extra, 0, len(P.LAND_PRICES) - 1)
    legal |= (
        (op == C.MARKET_OP_BUY_LAND)
        & (n_extra < len(P.LAND_PRICES))
        & (money >= _LAND_PRICES[land_index])
    )
    crop = jnp.clip(arg, 0, C.N_CROPS - 1)
    legal |= (op == C.MARKET_OP_BUY_SEED) & (money >= _SEED_COSTS[crop])
    animal = jnp.clip(arg, 0, C.N_ANIMALS - 1)
    legal |= (op == C.MARKET_OP_BUY_ANIMAL) & room & (money >= _ANIMAL_COSTS[animal])
    product = jnp.clip(arg, 0, C.N_PRODUCTS - 1)
    buy_price = market.market_price_one(product, state.market_inventory[product] - 1)
    legal |= (op == C.MARKET_OP_BUY_PRODUCT) & room & (money >= buy_price)
    legal |= (op == C.MARKET_OP_SELL) & (shed[product] > 0)
    return legal


def unit_requires_quantity(state: State, player: jnp.ndarray, unit: jnp.ndarray, op, arg):
    """選択したunit行動に数量決定が必要か返す。"""
    pos, _ = unit_fields(state, player, unit)
    tile_kind = state.tiles_kind[player, pos[1], pos[0]]
    tile_item = state.tiles_crop_or_animal[player, pos[1], pos[0]]
    animal_place = _animal_placement(tile_kind, tile_item, arg)
    return (op == C.FARMER_OP_PICKUP) | ((op == C.FARMER_OP_PLACE) & ~animal_place)


def unit_quantity_upper_bound(state: State, player: jnp.ndarray, unit: jnp.ndarray, op, arg):
    """数量head向けに共有納屋へ依存しない上限を返す。

    Args:
        state: ターン開始時の状態。
        player: 対象プレイヤー。
        unit: 対象unitの固定slot index。
        op: 選択したunit操作。
        arg: 選択した品目index。

    Returns:
        Executorで逐次クランプする前の数量上限。
    """
    _, inventory = unit_fields(state, player, unit)
    held = inventory[jnp.clip(arg, 0, C.N_SHED_ITEMS - 1)]
    bound = jnp.where(op == C.FARMER_OP_PICKUP, V.MAX_ACTION_QUANTITY, held)
    return jnp.clip(bound, 0, V.MAX_ACTION_QUANTITY)


def max_unit_quantity(
    state: State, player: jnp.ndarray, unit: jnp.ndarray, op, arg, shed_capacity: int = 100
):
    """選択済みunit行動の実行可能数量上限を返す(Executorが逐次stateに対して呼ぶ、厳密版)。"""
    _, inventory = unit_fields(state, player, unit)
    shed = state.shed[player]
    pickup = shed[jnp.clip(arg, 0, C.N_SHED_ITEMS - 1)]
    place = jnp.minimum(
        inventory[jnp.clip(arg, 0, C.N_SHED_ITEMS - 1)],
        jnp.maximum(shed_capacity - jnp.sum(shed), 0),
    )
    result = jnp.where(op == C.FARMER_OP_PICKUP, pickup, place)
    return jnp.clip(result, 0, V.MAX_ACTION_QUANTITY)


def max_market_quantity(
    state: State,
    player: jnp.ndarray,
    op,
    arg,
    *,
    shed_capacity: int = 100,
):
    """選択済み市場注文の実行可能数量上限を返す。"""
    money = state.money[player]
    shed = state.shed[player]
    room = jnp.maximum(shed_capacity - jnp.sum(shed), 0)
    crop = jnp.clip(arg, 0, C.N_CROPS - 1)
    animal = jnp.clip(arg, 0, C.N_ANIMALS - 1)
    product = jnp.clip(arg, 0, C.N_PRODUCTS - 1)

    simple = jnp.where(
        op == C.MARKET_OP_BUY_SEED,
        jnp.floor(money / _SEED_COSTS[crop]).astype(jnp.int32),
        jnp.where(
            op == C.MARKET_OP_BUY_ANIMAL,
            jnp.minimum(jnp.floor(money / _ANIMAL_COSTS[animal]).astype(jnp.int32), room),
            jnp.where(op == C.MARKET_OP_SELL, shed[product], 0),
        ),
    )

    def cond(carry):
        count, cash, held, inventory = carry
        price = market.market_price_one(product, inventory - 1)
        return (
            (count < V.MAX_ACTION_QUANTITY)
            & (cash >= price)
            & (jnp.sum(shed) - shed[product] + held < shed_capacity)
        )

    def body(carry):
        count, cash, held, inventory = carry
        price = market.market_price_one(product, inventory - 1)
        return count + 1, cash - price, held + 1, inventory - 1

    bought, *_ = jax.lax.while_loop(
        cond,
        body,
        (jnp.asarray(0), money, shed[product], state.market_inventory[product]),
    )
    result = jnp.where(op == C.MARKET_OP_BUY_PRODUCT, bought, simple)
    return jnp.clip(result, 0, V.MAX_ACTION_QUANTITY)


def commit_unit_action(
    state: State,
    player: jnp.ndarray,
    unit: jnp.ndarray,
    op: jnp.ndarray,
    arg: jnp.ndarray,
    quantity: jnp.ndarray,
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
) -> State:
    """選択した1ユニット行動を自己回帰用の仮Stateへ適用する。"""
    pos, inventory = unit_fields(state, player, unit)
    x, y = pos
    tile_fields = (
        state.tiles_kind,
        state.tiles_crop_or_animal,
        state.tiles_planted_or_placed_day,
        state.tiles_watered_or_fed_today,
        state.tiles_consecutive_unwatered_or_unfed,
        state.tiles_yield_units,
        state.tiles_max_lifespan_step,
        state.tiles_fertilized_until_day,
        state.tiles_cared_today,
        state.tiles_fertilizer_available,
        state.tiles_pending_care_bonus,
    )
    gathered = tuple(field[player, y, x] for field in tile_fields)
    result = apply_unit_action(
        op,
        jnp.maximum(arg, 0),
        quantity,
        pos,
        state.tiles_kind.shape[-1],
        *gathered,
        state.step // turns_per_day,
        turns_per_day,
        state.shed[player],
        inventory,
        shed_capacity,
        jnp.zeros((C.N_CROPS,), dtype=bool),
    )
    new_pos, new_tile_values, new_shed, new_inventory, seed_used = (
        result[0],
        result[1:12],
        result[12],
        result[13],
        result[14],
    )
    updated_tiles = tuple(
        field.at[player, y, x].set(value)
        for field, value in zip(tile_fields, new_tile_values, strict=True)
    )
    hand = jnp.maximum(unit - 1, 0)
    is_farmer = unit == 0
    farmer_pos = state.farmer_pos.at[player].set(
        jnp.where(is_farmer, new_pos, state.farmer_pos[player])
    )
    hands_pos = state.hands_pos.at[player, hand].set(
        jnp.where(is_farmer, state.hands_pos[player, hand], new_pos)
    )
    farmer_inventory = state.farmer_inventory.at[player].set(
        jnp.where(is_farmer, new_inventory, state.farmer_inventory[player])
    )
    hands_inventory = state.hands_inventory.at[player, hand].set(
        jnp.where(is_farmer, state.hands_inventory[player, hand], new_inventory)
    )
    return state._replace(
        tiles_kind=updated_tiles[0],
        tiles_crop_or_animal=updated_tiles[1],
        tiles_planted_or_placed_day=updated_tiles[2],
        tiles_watered_or_fed_today=updated_tiles[3],
        tiles_consecutive_unwatered_or_unfed=updated_tiles[4],
        tiles_yield_units=updated_tiles[5],
        tiles_max_lifespan_step=updated_tiles[6],
        tiles_fertilized_until_day=updated_tiles[7],
        tiles_cared_today=updated_tiles[8],
        tiles_fertilizer_available=updated_tiles[9],
        tiles_pending_care_bonus=updated_tiles[10],
        farmer_pos=farmer_pos,
        hands_pos=hands_pos,
        shed=state.shed.at[player].set(new_shed),
        farmer_inventory=farmer_inventory,
        hands_inventory=hands_inventory,
        seeds=state.seeds.at[player].set(state.seeds[player] - seed_used),
    )


def commit_market_action(
    state: State,
    player: jnp.ndarray,
    op: jnp.ndarray,
    arg: jnp.ndarray,
    quantity: jnp.ndarray,
    *,
    hire_mult: float = 1.0,
    shed_capacity: int = 100,
) -> State:
    """自分の市場注文だけを仮Stateへ適用する。相手同時注文は未知なので含めない。"""
    board_size = state.tiles_kind.shape[-1]
    money = state.money[player]
    hires = state.hires_today[player]
    hands_active = state.hands_active[player]
    hands_pos = state.hands_pos[player]
    unlocked = state.unlocked_quadrants[player]
    tile_kind = state.tiles_kind[player]
    seeds = state.seeds[player]
    shed = state.shed[player]

    money, hires, hands_active, hands_pos = market_orders.apply_hire(
        op,
        money,
        hires,
        hands_active,
        hands_pos,
        state.farmer_pos[player],
        board_size,
        hire_mult,
    )
    money, unlocked, tile_kind = market_orders.apply_buy_land(
        op, money, unlocked, tile_kind, board_size
    )
    money, seeds = market_orders.apply_buy_seed(
        op, jnp.clip(arg, 0, C.N_CROPS - 1), quantity, money, seeds
    )
    money, shed = market_orders.apply_buy_animal(
        op,
        jnp.clip(arg, 0, C.N_ANIMALS - 1),
        quantity,
        money,
        shed,
        shed_capacity,
    )
    money, shed, _, _, market_inventory = market_lockstep(
        op,
        jnp.clip(arg, 0, C.N_PRODUCTS - 1),
        quantity,
        money,
        shed,
        jnp.asarray(-1),
        jnp.asarray(0),
        jnp.asarray(0),
        jnp.asarray(0.0),
        jnp.zeros_like(shed),
        state.market_inventory,
        shed_capacity,
    )
    return state._replace(
        money=state.money.at[player].set(money),
        hires_today=state.hires_today.at[player].set(hires),
        hands_active=state.hands_active.at[player].set(hands_active),
        hands_pos=state.hands_pos.at[player].set(hands_pos),
        unlocked_quadrants=state.unlocked_quadrants.at[player].set(unlocked),
        tiles_kind=state.tiles_kind.at[player].set(tile_kind),
        seeds=state.seeds.at[player].set(seeds),
        shed=state.shed.at[player].set(shed),
        market_inventory=market_inventory,
    )
