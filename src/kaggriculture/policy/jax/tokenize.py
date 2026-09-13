"""JAX Stateを固定形状の方策トークンへ直接変換する。"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.jax import features as F
from kaggriculture.rules import constants as C
from kaggriculture.simulator import market
from kaggriculture.simulator.state import State


class EpisodeCounters(NamedTuple):
    """rollout中にGPU上で保持する自分視点の累積実績。"""

    produced: jnp.ndarray
    sold: jnp.ndarray
    estimated_bought_product: jnp.ndarray
    estimated_revenue: jnp.ndarray
    has_ever_sold: jnp.ndarray

    @classmethod
    def zeros(cls) -> "EpisodeCounters":
        values = jnp.zeros((C.N_PRODUCTS,), dtype=jnp.float32)
        return cls(values, values, values, values, jnp.zeros((C.N_PRODUCTS,), dtype=bool))


def _blank(n_tokens: int, width: int) -> F.DenseFeatures:
    return F.DenseFeatures(
        jnp.zeros((n_tokens, width), dtype=jnp.int32),
        jnp.zeros((n_tokens, width), dtype=jnp.float32),
    )


def _bucket(normalized: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(
        (normalized * V.N_MAGNITUDE_BUCKETS).astype(jnp.int32),
        0,
        V.N_MAGNITUDE_BUCKETS - 1,
    )


def _norm_log(values: jnp.ndarray, cap: float) -> jnp.ndarray:
    return jnp.log1p(jnp.maximum(values, 0)) / jnp.log1p(jnp.asarray(cap, jnp.float32))


def _count_token(
    counts: jnp.ndarray,
    entity_ids: jnp.ndarray,
    entity_table: range,
    magnitude_table: range,
    cap: float,
    width: int,
    continuous_table: range | None = None,
) -> F.DenseFeatures:
    n = counts.shape[0]
    stride = 3 if continuous_table is not None else 2
    token = _blank(1, width)
    slots = jnp.arange(n, dtype=jnp.int32) * stride
    present = counts > 0
    normalized = _norm_log(counts, cap)
    index = token.index.at[0, slots].set(entity_table.start + entity_ids)
    index = index.at[0, slots + 1].set(
        magnitude_table.start + entity_ids * V.N_MAGNITUDE_BUCKETS + _bucket(normalized)
    )
    value = token.value.at[0, slots].set(present.astype(jnp.float32))
    value = value.at[0, slots + 1].set(present.astype(jnp.float32))
    if continuous_table is not None:
        index = index.at[0, slots + 2].set(continuous_table.start + entity_ids)
        value = value.at[0, slots + 2].set(jnp.where(present, normalized, 0.0))
    return F.DenseFeatures(index, value)


def _board_tokens(state: State, owner: jnp.ndarray, day: jnp.ndarray) -> F.DenseFeatures:
    width = F.MAX_ENCODER_FEATURES
    kind = state.tiles_kind[owner].reshape(-1)
    item = state.tiles_crop_or_animal[owner].reshape(-1)
    placed_day = state.tiles_planted_or_placed_day[owner].reshape(-1)
    cared = state.tiles_watered_or_fed_today[owner].reshape(-1)
    uncared = state.tiles_consecutive_unwatered_or_unfed[owner].reshape(-1)
    yields = state.tiles_yield_units[owner].reshape(-1)
    lifespan = state.tiles_max_lifespan_step[owner].reshape(-1)
    fertilized = state.tiles_fertilized_until_day[owner].reshape(-1)
    animal_cared = state.tiles_cared_today[owner].reshape(-1)
    fertilizer_available = state.tiles_fertilizer_available[owner].reshape(-1)
    care_bonus = state.tiles_pending_care_bonus[owner].reshape(-1)

    n = kind.shape[0]
    rows = jnp.arange(n)
    token = _blank(n, width)
    index = token.index.at[rows, 0].set(V.TILE_KIND.start + kind)
    value = token.value.at[rows, 0].set(1.0)

    is_plant = kind == C.TILE_PLANT
    is_animal = ((kind == C.TILE_COOP) | (kind == C.TILE_PASTURE)) & (item >= 0)
    has_entity = is_plant | is_animal
    entity = jnp.where(is_plant, item, C.N_PRODUCTS + item)
    index = index.at[rows, 1].set(V.ENTITY_ITEM.start + jnp.maximum(entity, 0))
    value = value.at[rows, 1].set(has_entity.astype(jnp.float32))

    index = index.at[rows, 2].set(V.TILE_CARE_DONE_TODAY.start)
    value = value.at[rows, 2].set((has_entity & cared).astype(jnp.float32))
    index = index.at[rows, 3].set(V.TILE_FERTILIZED_ACTIVE.start)
    value = value.at[rows, 3].set((is_plant & (fertilized >= day)).astype(jnp.float32))
    index = index.at[rows, 4].set(V.TILE_CARED_TODAY.start)
    value = value.at[rows, 4].set((is_animal & animal_cared).astype(jnp.float32))
    index = index.at[rows, 5].set(V.TILE_FERTILIZER_AVAILABLE.start)
    value = value.at[rows, 5].set((is_animal & fertilizer_available).astype(jnp.float32))

    index = index.at[rows, 6].set(V.TILE_AGE.start)
    value = value.at[rows, 6].set(jnp.where(has_entity, (day - placed_day) / 30.0, 0.0))
    index = index.at[rows, 7].set(V.TILE_YIELD_UNITS.start)
    value = value.at[rows, 7].set(jnp.where(has_entity, yields / 6.0, 0.0))
    index = index.at[rows, 8].set(V.TILE_CONSECUTIVE_UNCARED.start)
    value = value.at[rows, 8].set(jnp.where(has_entity, uncared / 2.0, 0.0))
    index = index.at[rows, 9].set(V.TILE_PENDING_CARE_BONUS.start)
    value = value.at[rows, 9].set(jnp.where(is_animal, jnp.clip(care_bonus, 0, 10) / 10.0, 0.0))
    index = index.at[rows, 10].set(V.TILE_LIFESPAN_REMAINING.start)
    value = value.at[rows, 10].set(
        jnp.where(
            is_plant & (lifespan >= 0),
            jnp.clip(lifespan - state.step, 0, 60) / 60.0,
            0.0,
        )
    )

    size = state.tiles_kind.shape[-1]
    y, x = jnp.divmod(rows, size)
    farmer_pos = state.farmer_pos[owner]
    is_farmer = (x == farmer_pos[0]) & (y == farmer_pos[1])
    hand_match = (state.hands_pos[owner, :, 0, None] == x[None, :]) & (
        state.hands_pos[owner, :, 1, None] == y[None, :]
    )
    hand_count = jnp.sum(hand_match & state.hands_active[owner, :, None], axis=0)
    index = index.at[rows, 11].set(V.TILE_IS_FARMER.start)
    value = value.at[rows, 11].set(is_farmer.astype(jnp.float32))
    index = index.at[rows, 12].set(V.TILE_HAND_COUNT.start)
    value = value.at[rows, 12].set(jnp.clip(hand_count, 0, 8) / 8.0)
    return F.DenseFeatures(index, value)


def _player_token(state: State, owner: jnp.ndarray) -> F.DenseFeatures:
    token = _blank(1, F.MAX_ENCODER_FEATURES)
    money_norm = _norm_log(state.money[owner], 200_000)
    index = token.index.at[0, 0].set(V.PLAYER_MONEY.start)
    index = index.at[0, 1].set(V.PLAYER_MONEY_MAGNITUDE_BUCKET.start + _bucket(money_norm))
    index = index.at[0, 2].set(V.PLAYER_MONEY_CONTINUOUS.start)
    q_slots = jnp.arange(C.N_QUADRANTS) + 3
    index = index.at[0, q_slots].set(V.PLAYER_UNLOCKED_QUADRANT.start + jnp.arange(C.N_QUADRANTS))
    index = index.at[0, 7].set(V.PLAYER_HIRES_TODAY.start)
    value = token.value.at[0, :3].set(jnp.array([1.0, 1.0, money_norm]))
    value = value.at[0, q_slots].set(state.unlocked_quadrants[owner].astype(jnp.float32))
    value = value.at[0, 7].set(jnp.clip(state.hires_today[owner], 0, 16) / 16.0)
    return F.DenseFeatures(index, value)


def _counter_tokens(counters: EpisodeCounters) -> F.DenseFeatures:
    """累積実績をカテゴリ別の5トークンへ符号化する。"""
    parts = (
        (
            counters.produced,
            V.OWN_PRODUCED,
            V.OWN_PRODUCED_MAGNITUDE_BUCKET,
            V.OWN_PRODUCED_CONTINUOUS,
            1_000,
        ),
        (
            counters.sold,
            V.OWN_SOLD,
            V.OWN_SOLD_MAGNITUDE_BUCKET,
            V.OWN_SOLD_CONTINUOUS,
            1_000,
        ),
        (
            counters.estimated_bought_product,
            V.OWN_ESTIMATED_BOUGHT_PRODUCT,
            V.OWN_ESTIMATED_BOUGHT_PRODUCT_MAGNITUDE_BUCKET,
            V.OWN_ESTIMATED_BOUGHT_PRODUCT_CONTINUOUS,
            1_000,
        ),
        (
            counters.estimated_revenue,
            V.OWN_ESTIMATED_REVENUE,
            V.OWN_ESTIMATED_REVENUE_MAGNITUDE_BUCKET,
            V.OWN_ESTIMATED_REVENUE_CONTINUOUS,
            500_000,
        ),
    )
    entity_ids = jnp.arange(C.N_PRODUCTS)
    tokens = [
        _count_token(
            counts,
            entity_ids,
            entity_table,
            magnitude_table,
            cap,
            F.MAX_ENCODER_FEATURES,
            continuous_table,
        )
        for counts, entity_table, magnitude_table, continuous_table, cap in parts
    ]
    sold = _blank(1, F.MAX_ENCODER_FEATURES)
    slots = jnp.arange(C.N_PRODUCTS)
    sold_index = sold.index.at[0, slots].set(V.OWN_HAS_EVER_SOLD.start + slots)
    sold_value = sold.value.at[0, slots].set(counters.has_ever_sold.astype(jnp.float32))
    tokens.append(F.DenseFeatures(sold_index, sold_value))
    return F.DenseFeatures(
        jnp.concatenate([token.index for token in tokens]),
        jnp.concatenate([token.value for token in tokens]),
    )


def encode_observation(
    state: State,
    player: jnp.ndarray,
    counters: EpisodeCounters | None = None,
    turns_per_day: int = 24,
) -> F.DenseFeatures:
    """1局面・1プレイヤー分のactor入力を作る。`jax.vmap`可能。"""
    counters = EpisodeCounters.zeros() if counters is None else counters
    opponent = 1 - player
    day = state.step // turns_per_day
    hour = state.step % turns_per_day
    entity_products = jnp.arange(C.N_PRODUCTS)

    own_board = _board_tokens(state, player, day)
    opponent_board = _board_tokens(state, opponent, day)
    own_shed = _count_token(
        state.shed[player],
        jnp.arange(C.N_SHED_ITEMS),
        V.ENTITY_ITEM,
        V.ENTITY_MAGNITUDE_BUCKET,
        100,
        F.MAX_ENCODER_FEATURES,
    )
    own_seeds = _count_token(
        state.seeds[player],
        jnp.arange(C.N_CROPS),
        V.ENTITY_ITEM,
        V.ENTITY_MAGNITUDE_BUCKET,
        100,
        F.MAX_ENCODER_FEATURES,
    )
    inventory = state.farmer_inventory[player] + jnp.sum(state.hands_inventory[player], axis=0)
    own_inventory = _count_token(
        inventory,
        jnp.arange(C.N_SHED_ITEMS),
        V.ENTITY_ITEM,
        V.ENTITY_MAGNITUDE_BUCKET,
        100,
        F.MAX_ENCODER_FEATURES,
    )
    market_inventory = _count_token(
        state.market_inventory,
        entity_products,
        V.ENTITY_ITEM,
        V.ENTITY_MAGNITUDE_BUCKET,
        12_000,
        F.MAX_ENCODER_FEATURES,
    )
    prices = market.market_price(state.market_inventory)
    market_prices = _count_token(
        prices,
        entity_products,
        V.ENTITY_ITEM,
        V.ENTITY_MAGNITUDE_BUCKET,
        2_000,
        F.MAX_ENCODER_FEATURES,
    )

    town = _blank(1, F.MAX_ENCODER_FEATURES)
    shop_ids = jnp.arange(C.N_SHOPS)
    shop_present = state.town_shop_counts > 0
    shop_norm = state.town_shop_counts / 8.0
    town_index = town.index.at[0, 2 * shop_ids].set(V.TOWN_SHOP.start + shop_ids)
    town_index = town_index.at[0, 2 * shop_ids + 1].set(
        V.TOWN_SHOP_MAGNITUDE_BUCKET.start + shop_ids * V.N_MAGNITUDE_BUCKETS + _bucket(shop_norm)
    )
    town_value = town.value.at[0, 2 * shop_ids].set(shop_present.astype(jnp.float32))
    town_value = town_value.at[0, 2 * shop_ids + 1].set(shop_present.astype(jnp.float32))

    turn = _blank(1, F.MAX_ENCODER_FEATURES)
    turn_index = turn.index.at[0, 0].set(V.TURN_DAY.start)
    turn_index = turn_index.at[0, 1].set(V.TURN_HOUR.start)
    turn_value = turn.value.at[0, 0].set(day / 30.0)
    turn_value = turn_value.at[0, 1].set(hour / 24.0)

    features = (
        own_board,
        opponent_board,
        _player_token(state, player),
        _player_token(state, opponent),
        own_shed,
        own_seeds,
        own_inventory,
        market_inventory,
        market_prices,
        F.DenseFeatures(town_index, town_value),
        F.DenseFeatures(turn_index, turn_value),
        _counter_tokens(counters),
    )
    result = F.DenseFeatures(
        jnp.concatenate([item.index for item in features]),
        jnp.concatenate([item.value for item in features]),
    )
    if result.index.shape[0] != L.NUM_WORDS_ENCODER:
        raise AssertionError("encoder token layout mismatch")
    return result


def encode_privileged(
    state: State, player: jnp.ndarray
) -> tuple[F.DenseFeatures, jnp.ndarray, jnp.ndarray]:
    """1局面・1プレイヤー分の非対称critic入力を作る。"""
    owners = jnp.array([player, 1 - player])
    all_features = []
    all_positions = []
    all_padding = []
    for owner in owners:
        shed = _count_token(
            state.shed[owner],
            jnp.arange(C.N_SHED_ITEMS),
            V.ENTITY_ITEM,
            V.ENTITY_MAGNITUDE_BUCKET,
            100,
            F.MAX_PRIVILEGED_FEATURES,
        )
        seeds = _count_token(
            state.seeds[owner],
            jnp.arange(C.N_CROPS),
            V.ENTITY_ITEM,
            V.ENTITY_MAGNITUDE_BUCKET,
            100,
            F.MAX_PRIVILEGED_FEATURES,
        )
        inventories = jnp.concatenate(
            [state.farmer_inventory[owner][None], state.hands_inventory[owner]], axis=0
        )
        inventory_features = jax.vmap(
            lambda counts: _count_token(
                counts,
                jnp.arange(C.N_SHED_ITEMS),
                V.ENTITY_ITEM,
                V.ENTITY_MAGNITUDE_BUCKET,
                100,
                F.MAX_PRIVILEGED_FEATURES,
            )
        )(inventories)
        inventory_features = F.DenseFeatures(
            inventory_features.index[:, 0], inventory_features.value[:, 0]
        )
        positions = jnp.concatenate([state.farmer_pos[owner][None], state.hands_pos[owner]])
        position_ids = positions[:, 1] * V.BOARD_SIZE + positions[:, 0]
        padding = jnp.concatenate([jnp.array([False]), ~state.hands_active[owner]])
        position_ids = jnp.where(padding, L.NO_POSITION, position_ids)
        all_features.extend([shed, seeds, inventory_features])
        all_positions.append(jnp.concatenate([jnp.full((2,), L.NO_POSITION), position_ids]))
        all_padding.append(jnp.concatenate([jnp.zeros((2,), dtype=bool), padding]))
    features = F.DenseFeatures(
        jnp.concatenate([item.index for item in all_features]),
        jnp.concatenate([item.value for item in all_features]),
    )
    return features, jnp.concatenate(all_positions), jnp.concatenate(all_padding)
