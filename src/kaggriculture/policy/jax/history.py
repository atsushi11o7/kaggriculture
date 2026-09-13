"""確定行動からJAX上でエピソード累積実績を更新する。"""

import jax
import jax.numpy as jnp

from kaggriculture.policy.jax import actions as A
from kaggriculture.policy.jax.tokenize import EpisodeCounters
from kaggriculture.rules import constants as C
from kaggriculture.rules import game_params as P
from kaggriculture.simulator.action import Action
from kaggriculture.simulator.state import State

_CROP_PRODUCT = jnp.asarray(P.CROP_PRODUCT_IDX)
_ANIMAL_PRODUCT = jnp.asarray(P.ANIMAL_PRODUCT_IDX)
_FERTILIZER = C.PRODUCTS.index("FERTILIZER")


def update_player_counters(
    state: State,
    action: Action,
    player: jnp.ndarray,
    counters: EpisodeCounters,
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1.0,
) -> EpisodeCounters:
    """1局・1プレイヤーの確定行動を単独再生して累積値を更新する。"""

    def unit_step(unit, carry):
        shadow, produced = carry
        is_farmer = unit == 0
        hand = jnp.maximum(unit - 1, 0)
        op = jnp.where(is_farmer, action.farmer_op[player], action.hands_op[player, hand])
        arg = jnp.where(
            is_farmer, action.farmer_arg_idx[player], action.hands_arg_idx[player, hand]
        )
        quantity = jnp.where(is_farmer, action.farmer_n[player], action.hands_n[player, hand])
        pos, _ = A.unit_fields(shadow, player, unit)
        x, y = pos
        kind = shadow.tiles_kind[player, y, x]
        item = shadow.tiles_crop_or_animal[player, y, x]
        units = shadow.tiles_yield_units[player, y, x]
        crop_product = _CROP_PRODUCT[jnp.clip(item, 0, C.N_CROPS - 1)]
        animal_product = _ANIMAL_PRODUCT[jnp.clip(item, 0, C.N_ANIMALS - 1)]
        product = jnp.where(kind == C.TILE_PLANT, crop_product, animal_product)
        harvested = (op == C.FARMER_OP_HARVEST) & (units > 0)
        fertilizer = op == C.FARMER_OP_COLLECT_FERTILIZER
        product = jnp.where(fertilizer, _FERTILIZER, product)
        amount = jnp.where(harvested, units, jnp.where(fertilizer, 1, 0))
        produced += jax.nn.one_hot(product, C.N_PRODUCTS, dtype=jnp.float32) * amount
        shadow = A.commit_unit_action(
            shadow,
            player,
            unit,
            op,
            arg,
            quantity,
            turns_per_day=turns_per_day,
            shed_capacity=shed_capacity,
        )
        return shadow, produced

    n_units = 1 + jnp.sum(state.hands_active[player])
    shadow, produced = jax.lax.fori_loop(
        0, n_units, unit_step, (state, jnp.zeros((C.N_PRODUCTS,), dtype=jnp.float32))
    )

    def market_step(slot, carry):
        shadow, sold, bought, revenue = carry
        op = action.market_op[player, slot]
        arg = action.market_arg_idx[player, slot]
        quantity = action.market_n[player, slot]
        product = jnp.clip(arg, 0, C.N_PRODUCTS - 1)
        before_shed = shadow.shed[player, product]
        before_money = shadow.money[player]

        def commit(current):
            return A.commit_market_action(
                current,
                player,
                op,
                arg,
                quantity,
                hire_mult=hire_mult,
                shed_capacity=shed_capacity,
            )

        updated = jax.lax.cond(op >= 0, commit, lambda current: current, shadow)
        after_shed = updated.shed[player, product]
        delta_sold = jnp.where(op == C.MARKET_OP_SELL, before_shed - after_shed, 0)
        delta_bought = jnp.where(op == C.MARKET_OP_BUY_PRODUCT, after_shed - before_shed, 0)
        delta_revenue = jnp.where(op == C.MARKET_OP_SELL, updated.money[player] - before_money, 0)
        sold += jax.nn.one_hot(product, C.N_PRODUCTS) * delta_sold
        bought += jax.nn.one_hot(product, C.N_PRODUCTS) * delta_bought
        revenue += jax.nn.one_hot(product, C.N_PRODUCTS) * delta_revenue
        return updated, sold, bought, revenue

    zero = jnp.zeros((C.N_PRODUCTS,), dtype=jnp.float32)
    _, sold, bought, revenue = jax.lax.fori_loop(
        0, C.MAX_MARKET_ORDERS, market_step, (shadow, zero, zero, zero)
    )
    return EpisodeCounters(
        produced=counters.produced + produced,
        sold=counters.sold + sold,
        estimated_bought_product=counters.estimated_bought_product + bought,
        estimated_revenue=counters.estimated_revenue + revenue,
        has_ever_sold=counters.has_ever_sold | (sold > 0),
    )


def update_counters(
    states: State,
    actions: Action,
    counters: EpisodeCounters,
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1.0,
) -> EpisodeCounters:
    """batch内の両プレイヤーの累積値を更新する。"""

    def game(state, action, game_counters):
        player0 = jax.tree.map(lambda value: value[0], game_counters)
        player1 = jax.tree.map(lambda value: value[1], game_counters)
        updated0 = update_player_counters(
            state,
            action,
            jnp.asarray(0),
            player0,
            turns_per_day=turns_per_day,
            shed_capacity=shed_capacity,
            hire_mult=hire_mult,
        )
        updated1 = update_player_counters(
            state,
            action,
            jnp.asarray(1),
            player1,
            turns_per_day=turns_per_day,
            shed_capacity=shed_capacity,
            hire_mult=hire_mult,
        )
        return jax.tree.map(lambda a, b: jnp.stack([a, b]), updated0, updated1)

    return jax.vmap(game)(states, actions, counters)


def zeros(batch_size: int) -> EpisodeCounters:
    """`[env, player, product]`形のゼロcounterを作る。"""
    value = jnp.zeros((batch_size, 2, C.N_PRODUCTS), dtype=jnp.float32)
    return EpisodeCounters(value, value, value, value, value.astype(bool))
