"""並列に選んだintentをシミュレータ用の合法な固定shape Actionへ変換する。"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture.policy.jax import actions as A
from kaggriculture.policy.jax.types import ExecutorStats, Intent
from kaggriculture.rules import constants as C
from kaggriculture.simulator.action import Action
from kaggriculture.simulator.state import State


def _execute_one(
    state: State,
    player: jnp.ndarray,
    intent: Intent,
    *,
    turns_per_day: int,
    shed_capacity: int,
    hire_mult: float,
):
    """1プレイヤーのintentを整形する。戦略的な並べ替えは行わない。"""
    unit_choice = intent.unit
    unit_op = A.UNIT_CANDIDATES.op[unit_choice]
    unit_arg = jnp.maximum(A.UNIT_CANDIDATES.arg[unit_choice], 0)
    units = jnp.arange(C.MAX_HANDS + 1, dtype=jnp.int32)
    unit_active = jnp.concatenate([jnp.ones((1,), dtype=bool), state.hands_active[player]], axis=0)
    needs_quantity = jax.vmap(A.unit_requires_quantity, in_axes=(None, None, 0, 0, 0))(
        state, player, units, unit_op, unit_arg
    )
    requested_unit_n = jnp.where(needs_quantity, intent.unit_quantity + 1, 1)
    maximum_unit_n = jax.vmap(
        lambda unit, op, arg: A.max_unit_quantity(
            state, player, unit, op, arg, shed_capacity=shed_capacity
        )
    )(units, unit_op, unit_arg)
    unit_n = jnp.where(
        needs_quantity, jnp.minimum(requested_unit_n, jnp.maximum(maximum_unit_n, 1)), 1
    )
    clamped_unit = jnp.sum(
        (unit_active & needs_quantity & (unit_n != requested_unit_n)).astype(jnp.int32)
    )
    plant = unit_active & (unit_op == C.FARMER_OP_PLANT)
    demand = jnp.sum(jax.nn.one_hot(unit_arg, C.N_CROPS, dtype=jnp.int32) * plant[:, None], axis=0)
    blocked = demand > state.seeds[player]
    blocked_units = plant & blocked[jnp.clip(unit_arg, 0, C.N_CROPS - 1)]
    blocked_plants = jnp.sum(blocked_units.astype(jnp.int32))
    unit_op = jnp.where(blocked_units, C.FARMER_OP_PASS, unit_op)
    unit_op = jnp.where(unit_active, unit_op, C.FARMER_OP_PASS)
    unit_arg = jnp.where(unit_active, unit_arg, 0)
    unit_n = jnp.where(unit_active, unit_n, 1)

    market_op = jnp.full((C.MAX_MARKET_ORDERS,), -1, dtype=jnp.int32)
    market_arg = jnp.zeros((C.MAX_MARKET_ORDERS,), dtype=jnp.int32)
    market_n = jnp.zeros((C.MAX_MARKET_ORDERS,), dtype=jnp.int32)

    def market_step(carry, slot):
        shadow, active, ops, args, quantities, invalid, clamped, ignored = carry
        choice = intent.market[slot]
        proposed_op = A.MARKET_CANDIDATES.op[choice]
        proposed_arg = A.MARKET_CANDIDATES.arg[choice]
        is_stop = proposed_op == A.MARKET_STOP
        is_wait = proposed_op == A.MARKET_WAIT
        ignored = ignored + (~active).astype(jnp.int32)

        legal_mask = A.legal_market_mask(
            shadow, player, hire_mult=hire_mult, shed_capacity=shed_capacity
        )
        legal = legal_mask[choice]
        quantity_op = (
            (proposed_op == C.MARKET_OP_BUY_SEED)
            | (proposed_op == C.MARKET_OP_BUY_PRODUCT)
            | (proposed_op == C.MARKET_OP_BUY_ANIMAL)
            | (proposed_op == C.MARKET_OP_SELL)
        )
        requested = jnp.where(quantity_op, intent.market_quantity[slot] + 1, 1)
        maximum = A.max_market_quantity(
            shadow, player, proposed_op, proposed_arg, shed_capacity=shed_capacity
        )
        executable = active & ~is_stop & ~is_wait & legal & (~quantity_op | (maximum > 0))
        quantity = jnp.where(quantity_op, jnp.minimum(requested, maximum), 1)
        invalid = invalid + (active & ~is_stop & ~is_wait & ~executable).astype(jnp.int32)
        clamped = clamped + (executable & quantity_op & (quantity != requested)).astype(jnp.int32)

        # WAITは元環境と同じ数量0のSELLとして表す。slot位置は維持される。
        write_wait = active & is_wait
        write_regular = executable
        written_op = jnp.where(
            write_wait, C.MARKET_OP_SELL, jnp.where(write_regular, proposed_op, -1)
        )
        written_arg = jnp.where(write_wait, C.PRODUCTS.index("WHEAT"), jnp.maximum(proposed_arg, 0))
        written_n = jnp.where(write_regular, quantity, 0)
        ops = ops.at[slot].set(written_op)
        args = args.at[slot].set(written_arg)
        quantities = quantities.at[slot].set(written_n)
        shadow = jax.lax.cond(
            write_regular,
            lambda s: A.commit_market_action(
                s,
                player,
                proposed_op,
                proposed_arg,
                quantity,
                hire_mult=hire_mult,
                shed_capacity=shed_capacity,
            ),
            lambda s: s,
            shadow,
        )
        active = active & ~is_stop
        return (shadow, active, ops, args, quantities, invalid, clamped, ignored), None

    initial = (
        state,
        jnp.asarray(True),
        market_op,
        market_arg,
        market_n,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
    )
    final, _ = jax.lax.scan(market_step, initial, jnp.arange(C.MAX_MARKET_ORDERS))
    _, _, market_op, market_arg, market_n, invalid, clamped, ignored = final
    action = (
        unit_op[0],
        unit_arg[0],
        unit_n[0],
        unit_op[1:],
        unit_arg[1:],
        unit_n[1:],
        market_op,
        market_arg,
        market_n,
    )
    return action, ExecutorStats(
        jnp.asarray(0, jnp.int32),
        clamped_unit,
        blocked_plants,
        invalid,
        clamped,
        ignored,
    )


def execute(
    states: State,
    players: jnp.ndarray,
    intent: Intent,
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1.0,
) -> tuple[Action, ExecutorStats]:
    """batch内の各プレイヤーintentをJAX上で整形する。"""
    action_fields, stats = jax.vmap(
        lambda state, player, selected: _execute_one(
            state,
            player,
            selected,
            turns_per_day=turns_per_day,
            shed_capacity=shed_capacity,
            hire_mult=hire_mult,
        )
    )(states, players, intent)
    return Action(*action_fields), stats
