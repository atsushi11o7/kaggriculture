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
    unit_active = jnp.concatenate([jnp.ones((1,), dtype=bool), state.hands_active[player]], axis=0)

    # PLANTは元シミュレータと同様に全unitの需要を一括判定する。
    plant = unit_active & (unit_op == C.FARMER_OP_PLANT)
    demand = jnp.sum(jax.nn.one_hot(unit_arg, C.N_CROPS, dtype=jnp.int32) * plant[:, None], axis=0)
    blocked = demand > state.seeds[player]
    blocked_units = plant & blocked[jnp.clip(unit_arg, 0, C.N_CROPS - 1)]
    blocked_plants = jnp.sum(blocked_units.astype(jnp.int32))
    unit_op = jnp.where(blocked_units, C.FARMER_OP_PASS, unit_op)
    unit_op = jnp.where(unit_active, unit_op, C.FARMER_OP_PASS)
    unit_arg = jnp.where(unit_active, unit_arg, 0)

    def apply_shed_action(shed, inventory, op, arg, quantity, needs_quantity):
        """数量解決に必要な納屋の変化だけを適用する。"""
        item = jnp.clip(arg, 0, C.N_SHED_ITEMS - 1)
        pickup = op == C.FARMER_OP_PICKUP
        place = (op == C.FARMER_OP_PLACE) & needs_quantity
        shed = shed.at[item].add(jnp.where(pickup, -jnp.minimum(quantity, shed[item]), 0))
        room = jnp.maximum(shed_capacity - jnp.sum(shed), 0)
        placed = jnp.minimum(quantity, jnp.minimum(inventory[item], room))
        shed = shed.at[item].add(jnp.where(place, placed, 0))

        def apply_drop(current):
            def step(value, index):
                added = jnp.minimum(
                    inventory[index], jnp.maximum(shed_capacity - jnp.sum(value), 0)
                )
                return value.at[index].add(added), None

            return jax.lax.scan(step, current, jnp.arange(C.N_SHED_ITEMS))[0]

        return jax.lax.cond(op == C.FARMER_OP_DROP, apply_drop, lambda value: value, shed)

    # unit間で数量解決に影響する共有状態は納屋だけなので、State全体は複製しない。
    def unit_step(carry, i):
        shed, unit_n, invalid, clamped = carry
        op_i = unit_op[i]
        arg_i = unit_arg[i]
        active_i = unit_active[i]
        _, inventory_i = A.unit_fields(state, player, i)
        needs_i = A.unit_requires_quantity(state, player, i, op_i, arg_i)
        item_i = jnp.clip(arg_i, 0, C.N_SHED_ITEMS - 1)
        room_i = jnp.maximum(shed_capacity - jnp.sum(shed), 0)
        exact_max_i = jnp.where(
            op_i == C.FARMER_OP_PICKUP,
            shed[item_i],
            jnp.minimum(inventory_i[item_i], room_i),
        )
        exact_max_i = jnp.clip(exact_max_i, 0, A.QUANTITY_INDEX.shape[0])
        requested_i = jnp.where(needs_i, intent.unit_quantity[i] + 1, 1)
        quantity_i = jnp.where(needs_i, jnp.minimum(requested_i, jnp.maximum(exact_max_i, 1)), 1)
        drop_ineffective = (
            active_i & (op_i == C.FARMER_OP_DROP) & (room_i <= 0) & jnp.any(inventory_i > 0)
        )
        quantity_ineffective = active_i & needs_i & (exact_max_i == 0)
        invalid = invalid + (quantity_ineffective | drop_ineffective).astype(jnp.int32)
        clamped = clamped + (
            active_i & needs_i & (exact_max_i > 0) & (quantity_i != requested_i)
        ).astype(jnp.int32)
        unit_n = unit_n.at[i].set(jnp.where(active_i, quantity_i, 1))
        shed = jax.lax.cond(
            active_i,
            lambda value: apply_shed_action(value, inventory_i, op_i, arg_i, quantity_i, needs_i),
            lambda value: value,
            shed,
        )
        return (shed, unit_n, invalid, clamped), None

    unit_init = (
        state.shed[player],
        jnp.ones((C.MAX_HANDS + 1,), dtype=jnp.int32),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
    )
    (_, unit_n, invalid_unit, clamped_unit), _ = jax.lax.scan(
        unit_step, unit_init, jnp.arange(C.MAX_HANDS + 1, dtype=jnp.int32)
    )

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
        invalid_unit,
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
