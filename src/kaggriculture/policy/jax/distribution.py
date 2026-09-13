"""JAX/XLA上で完結する自己回帰方策分布。"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax.core import freeze

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.jax import actions as A
from kaggriculture.policy.jax import features as F
from kaggriculture.policy.jax import model as M
from kaggriculture.policy.jax import tokenize as T
from kaggriculture.rules import constants as C
from kaggriculture.simulator.action import Action
from kaggriculture.simulator.state import State

_UNIT_OP = 0
_UNIT_QUANTITY = 1
_MARKET_OP = 2
_MARKET_QUANTITY = 3
_DONE = 4
_MAX_CANDIDATES = V.MAX_ACTION_QUANTITY


class _PlayerAction(NamedTuple):
    farmer_op: jnp.ndarray
    farmer_arg_idx: jnp.ndarray
    farmer_n: jnp.ndarray
    hands_op: jnp.ndarray
    hands_arg_idx: jnp.ndarray
    hands_n: jnp.ndarray
    market_op: jnp.ndarray
    market_arg_idx: jnp.ndarray
    market_n: jnp.ndarray


class _DecodeCarry(NamedTuple):
    shadow: State
    action: _PlayerAction
    phase: jnp.ndarray
    unit: jnp.ndarray
    n_units: jnp.ndarray
    market_slot: jnp.ndarray
    pending_op: jnp.ndarray
    pending_arg: jnp.ndarray
    previous_index: jnp.ndarray
    previous_value: jnp.ndarray
    log_prob: jnp.ndarray
    entropy: jnp.ndarray
    num_decisions: jnp.ndarray
    keys: jnp.ndarray
    cache: dict


class SelfPlayOutput(NamedTuple):
    """両プレイヤーを同時生成したPPO rollout出力。"""

    action: Action
    value: jnp.ndarray
    log_prob: jnp.ndarray
    entropy: jnp.ndarray
    num_decisions: jnp.ndarray
    choices: jnp.ndarray
    decision_mask: jnp.ndarray
    token_log_prob: jnp.ndarray
    token_entropy: jnp.ndarray


class EvaluationOutput(NamedTuple):
    """保存済み候補列を現在の重みで再評価した結果。"""

    value: jnp.ndarray
    log_prob: jnp.ndarray
    entropy: jnp.ndarray
    num_decisions: jnp.ndarray
    token_log_prob: jnp.ndarray
    token_entropy: jnp.ndarray


class PolicyOutput(NamedTuple):
    """1プレイヤー分の複合行動とPPO収集値。"""

    action: _PlayerAction
    value: jnp.ndarray
    log_prob: jnp.ndarray
    entropy: jnp.ndarray
    num_decisions: jnp.ndarray
    choices: jnp.ndarray
    decision_mask: jnp.ndarray
    token_log_prob: jnp.ndarray
    token_entropy: jnp.ndarray


def state_values(
    model: M.PolicyValueNet,
    variables: dict,
    states: State,
    counters: T.EpisodeCounters | None = None,
    turns_per_day: int = 24,
) -> jnp.ndarray:
    """batch内の両プレイヤーの価値を一度に計算する。"""
    batch_size = states.step.shape[0]
    doubled_states = jax.tree.map(lambda value: jnp.concatenate([value, value]), states)
    players = jnp.concatenate(
        [jnp.zeros((batch_size,), dtype=jnp.int32), jnp.ones((batch_size,), dtype=jnp.int32)]
    )
    doubled_counters = None
    if counters is not None:
        doubled_counters = jax.tree.map(
            lambda value: jnp.concatenate([value[:, 0], value[:, 1]]), counters
        )
    if doubled_counters is None:
        zero = jnp.zeros((2 * batch_size, C.N_PRODUCTS), dtype=jnp.float32)
        doubled_counters = T.EpisodeCounters(zero, zero, zero, zero, zero.astype(bool))
    encoder = jax.vmap(
        lambda state, player, counter: T.encode_observation(state, player, counter, turns_per_day)
    )(doubled_states, players, doubled_counters)
    memory = model.apply(variables, encoder.index, encoder.value, method=model.encode)
    privileged = None
    if model.config.use_asymmetric_critic:
        features, positions, padding = jax.vmap(T.encode_privileged)(doubled_states, players)
        privileged = model.apply(
            variables,
            features.index,
            features.value,
            positions,
            padding,
            method=model.encode_privileged,
        )
    values = model.apply(variables, memory, privileged, method=model.get_value)
    return jnp.stack([values[:batch_size], values[batch_size:]], axis=1)


def combine_player_actions(player0: _PlayerAction, player1: _PlayerAction) -> Action:
    """両プレイヤーの出力をシミュレータ用Actionへまとめる。"""
    return Action(*(jnp.stack([a, b], axis=1) for a, b in zip(player0, player1, strict=True)))


def init_decode_cache(
    model: M.PolicyValueNet,
    key: jax.Array,
    batch_size: int,
    encoder_len: int = L.NUM_WORDS_ENCODER + 1,
) -> dict:
    """指定batch size用のゼロ初期化KV cacheを作る。"""
    d_model = model.config.d_model
    memory = jnp.zeros((batch_size, encoder_len, d_model), dtype=jnp.float32)
    index = jnp.zeros((batch_size, L.MAX_DECODE_LEN, F.MAX_DECODER_FEATURES), dtype=jnp.int32)
    value = jnp.zeros_like(index, dtype=jnp.float32)
    positions = jnp.zeros((batch_size, L.MAX_DECODE_LEN), dtype=jnp.int32)
    board_positions = jnp.full_like(positions, L.NO_POSITION)
    variables = model.init(
        key,
        memory,
        index,
        value,
        positions,
        board_positions,
        method=model.decode_step,
    )
    return jax.tree.map(jnp.zeros_like, variables["cache"])


def _empty_action(batch_size: int) -> _PlayerAction:
    return _PlayerAction(
        farmer_op=jnp.full((batch_size,), C.FARMER_OP_PASS, dtype=jnp.int32),
        farmer_arg_idx=jnp.zeros((batch_size,), dtype=jnp.int32),
        farmer_n=jnp.ones((batch_size,), dtype=jnp.int32),
        hands_op=jnp.full((batch_size, C.MAX_HANDS), C.FARMER_OP_PASS, dtype=jnp.int32),
        hands_arg_idx=jnp.zeros((batch_size, C.MAX_HANDS), dtype=jnp.int32),
        hands_n=jnp.ones((batch_size, C.MAX_HANDS), dtype=jnp.int32),
        market_op=jnp.full((batch_size, C.MAX_MARKET_ORDERS), -1, dtype=jnp.int32),
        market_arg_idx=jnp.zeros((batch_size, C.MAX_MARKET_ORDERS), dtype=jnp.int32),
        market_n=jnp.zeros((batch_size, C.MAX_MARKET_ORDERS), dtype=jnp.int32),
    )


def _unit_inventory(state: State, player, unit):
    hand = jnp.maximum(unit - 1, 0)
    return jnp.where(
        unit == 0,
        state.farmer_inventory[player],
        state.hands_inventory[player, hand],
    )


def _decoder_token(carry: _DecodeCarry, players: jnp.ndarray):
    batch_size = carry.phase.shape[0]
    index = jnp.zeros((batch_size, F.MAX_DECODER_FEATURES), dtype=jnp.int32)
    value = jnp.zeros((batch_size, F.MAX_DECODER_FEATURES), dtype=jnp.float32)
    index = index.at[:, : F.MAX_CANDIDATE_FEATURES].set(carry.previous_index)
    value = value.at[:, : F.MAX_CANDIDATE_FEATURES].set(carry.previous_value)

    inventories = jax.vmap(_unit_inventory)(carry.shadow, players, carry.unit)
    item_ids = jnp.arange(C.N_SHED_ITEMS)
    present = inventories > 0
    normalized = jnp.log1p(jnp.maximum(inventories, 0)) / jnp.log1p(100.0)
    buckets = jnp.clip(
        (normalized * V.N_MAGNITUDE_BUCKETS).astype(jnp.int32),
        0,
        V.N_MAGNITUDE_BUCKETS - 1,
    )
    is_unit = carry.phase == _UNIT_OP
    index = index.at[:, 2].set(V.UNIT_CONTEXT_INVENTORY.start)
    value = value.at[:, 2].set(is_unit.astype(jnp.float32))
    slots = 3 + 2 * item_ids
    index = index.at[:, slots].set(V.ENTITY_ITEM.start + item_ids)
    index = index.at[:, slots + 1].set(
        V.ENTITY_MAGNITUDE_BUCKET.start + item_ids * V.N_MAGNITUDE_BUCKETS + buckets
    )
    value = value.at[:, slots].set((present & is_unit[:, None]).astype(jnp.float32))
    value = value.at[:, slots + 1].set((present & is_unit[:, None]).astype(jnp.float32))

    unit_op_position = jnp.where(
        carry.unit == 0, L.FARMER_POSITION, L.HAND_BASE + 2 * (carry.unit - 1)
    )
    unit_quantity_position = unit_op_position + 1
    market_position = L.MARKET_BASE + 2 * carry.market_slot
    position = jnp.select(
        [carry.phase == _UNIT_OP, carry.phase == _UNIT_QUANTITY, carry.phase == _MARKET_OP],
        [unit_op_position, unit_quantity_position, market_position],
        default=market_position + 1,
    )
    positions = jax.vmap(lambda state, player, unit: A.unit_fields(state, player, unit)[0])(
        carry.shadow, players, carry.unit
    )
    board_position = positions[:, 1] * V.BOARD_SIZE + positions[:, 0]
    board_position = jnp.where(carry.phase == _UNIT_OP, board_position, L.NO_POSITION)
    return index[:, None], value[:, None], position[:, None], board_position[:, None]


def _pad_table(table: A.CandidateTable):
    n = table.op.shape[0]
    pad = _MAX_CANDIDATES - n
    return (
        jnp.pad(table.index, ((0, pad), (0, 0))),
        jnp.pad(table.value, ((0, pad), (0, 0))),
    )


_UNIT_INDEX, _UNIT_VALUE = _pad_table(A.UNIT_CANDIDATES)
_MARKET_INDEX, _MARKET_VALUE = _pad_table(A.MARKET_CANDIDATES)


def _candidate_batch(
    carry: _DecodeCarry,
    players: jnp.ndarray,
    turns_per_day: int,
    shed_capacity: int,
    hire_mult: float,
):
    unit_mask = jax.vmap(
        lambda state, player, unit: A.legal_unit_mask(
            state,
            player,
            unit,
            turns_per_day=turns_per_day,
            shed_capacity=shed_capacity,
        )
    )(carry.shadow, players, carry.unit)
    market_mask = jax.vmap(
        lambda state, player: A.legal_market_mask(
            state, player, hire_mult=hire_mult, shed_capacity=shed_capacity
        )
    )(carry.shadow, players)
    unit_mask = jnp.pad(unit_mask, ((0, 0), (0, _MAX_CANDIDATES - unit_mask.shape[1])))
    market_mask = jnp.pad(market_mask, ((0, 0), (0, _MAX_CANDIDATES - market_mask.shape[1])))
    quantity_max = jax.vmap(
        lambda state, player, unit, op, arg, phase: jnp.where(
            phase == _UNIT_QUANTITY,
            A.max_unit_quantity(state, player, unit, op, arg, shed_capacity=shed_capacity),
            A.max_market_quantity(state, player, op, arg, shed_capacity=shed_capacity),
        )
    )(
        carry.shadow,
        players,
        carry.unit,
        carry.pending_op,
        carry.pending_arg,
        carry.phase,
    )
    quantity_mask = jnp.arange(_MAX_CANDIDATES)[None] < quantity_max[:, None]
    is_unit = carry.phase == _UNIT_OP
    is_market = carry.phase == _MARKET_OP
    index = jnp.where(
        is_unit[:, None, None],
        _UNIT_INDEX[None],
        jnp.where(is_market[:, None, None], _MARKET_INDEX[None], A.QUANTITY_INDEX[None]),
    )
    value = jnp.where(
        is_unit[:, None, None],
        _UNIT_VALUE[None],
        jnp.where(is_market[:, None, None], _MARKET_VALUE[None], A.QUANTITY_VALUE[None]),
    )
    mask = jnp.where(
        is_unit[:, None],
        unit_mask,
        jnp.where(is_market[:, None], market_mask, quantity_mask),
    )
    done = carry.phase == _DONE
    mask = jnp.where(done[:, None], jnp.arange(_MAX_CANDIDATES)[None] == 0, mask)
    return index, value, mask


def _set_unit(action: _PlayerAction, unit, op, arg, quantity) -> _PlayerAction:
    is_farmer = unit == 0
    hand = jnp.maximum(unit - 1, 0)
    return action._replace(
        farmer_op=jnp.where(is_farmer, op, action.farmer_op),
        farmer_arg_idx=jnp.where(is_farmer, jnp.maximum(arg, 0), action.farmer_arg_idx),
        farmer_n=jnp.where(is_farmer, quantity, action.farmer_n),
        hands_op=action.hands_op.at[hand].set(jnp.where(is_farmer, action.hands_op[hand], op)),
        hands_arg_idx=action.hands_arg_idx.at[hand].set(
            jnp.where(is_farmer, action.hands_arg_idx[hand], jnp.maximum(arg, 0))
        ),
        hands_n=action.hands_n.at[hand].set(jnp.where(is_farmer, action.hands_n[hand], quantity)),
    )


def _set_market(action: _PlayerAction, slot, op, arg, quantity) -> _PlayerAction:
    return action._replace(
        market_op=action.market_op.at[slot].set(op),
        market_arg_idx=action.market_arg_idx.at[slot].set(jnp.maximum(arg, 0)),
        market_n=action.market_n.at[slot].set(quantity),
    )


def _advance_unit(unit, n_units):
    next_unit = unit + 1
    return next_unit, jnp.where(next_unit < n_units, _UNIT_OP, _MARKET_OP)


def _advance_market(slot):
    next_slot = slot + 1
    return next_slot, jnp.where(next_slot < C.MAX_MARKET_ORDERS, _MARKET_OP, _DONE)


def _transition_one(
    state,
    action,
    phase,
    unit,
    n_units,
    market_slot,
    pending_op,
    pending_arg,
    choice,
    player,
    turns_per_day,
    shed_capacity,
    hire_mult,
):
    def unit_op(_):
        op = A.UNIT_CANDIDATES.op[choice]
        arg = A.UNIT_CANDIDATES.arg[choice]
        needs_quantity = A.unit_requires_quantity(state, player, unit, op, arg)

        def defer(_):
            return state, action, _UNIT_QUANTITY, unit, market_slot, op, arg

        def commit(_):
            new_action = _set_unit(action, unit, op, arg, jnp.asarray(1))
            new_state = A.commit_unit_action(
                state,
                player,
                unit,
                op,
                arg,
                jnp.asarray(1),
                turns_per_day=turns_per_day,
                shed_capacity=shed_capacity,
            )
            next_unit, next_phase = _advance_unit(unit, n_units)
            return new_state, new_action, next_phase, next_unit, market_slot, op, arg

        return jax.lax.cond(needs_quantity, defer, commit, operand=None)

    def unit_quantity(_):
        quantity = choice + 1
        new_action = _set_unit(action, unit, pending_op, pending_arg, quantity)
        new_state = A.commit_unit_action(
            state,
            player,
            unit,
            pending_op,
            pending_arg,
            quantity,
            turns_per_day=turns_per_day,
            shed_capacity=shed_capacity,
        )
        next_unit, next_phase = _advance_unit(unit, n_units)
        return new_state, new_action, next_phase, next_unit, market_slot, pending_op, pending_arg

    def market_op(_):
        op = A.MARKET_CANDIDATES.op[choice]
        arg = A.MARKET_CANDIDATES.arg[choice]
        is_stop = op == A.MARKET_STOP
        is_wait = op == A.MARKET_WAIT
        needs_quantity = (
            (op == C.MARKET_OP_BUY_SEED)
            | (op == C.MARKET_OP_BUY_PRODUCT)
            | (op == C.MARKET_OP_BUY_ANIMAL)
            | (op == C.MARKET_OP_SELL)
        )

        def stop(_):
            return state, action, _DONE, unit, market_slot, op, arg

        def wait(_):
            new_action = _set_market(
                action,
                market_slot,
                jnp.asarray(C.MARKET_OP_SELL),
                jnp.asarray(C.PRODUCTS.index("WHEAT")),
                jnp.asarray(0),
            )
            next_slot, next_phase = _advance_market(market_slot)
            return state, new_action, next_phase, unit, next_slot, op, arg

        def regular(_):
            def defer(_):
                return state, action, _MARKET_QUANTITY, unit, market_slot, op, arg

            def commit(_):
                new_action = _set_market(action, market_slot, op, arg, jnp.asarray(1))
                new_state = A.commit_market_action(
                    state,
                    player,
                    op,
                    arg,
                    jnp.asarray(1),
                    hire_mult=hire_mult,
                    shed_capacity=shed_capacity,
                )
                next_slot, next_phase = _advance_market(market_slot)
                return new_state, new_action, next_phase, unit, next_slot, op, arg

            return jax.lax.cond(needs_quantity, defer, commit, operand=None)

        return jax.lax.cond(
            is_stop,
            stop,
            lambda _: jax.lax.cond(is_wait, wait, regular, operand=None),
            operand=None,
        )

    def market_quantity(_):
        quantity = choice + 1
        new_action = _set_market(action, market_slot, pending_op, pending_arg, quantity)
        new_state = A.commit_market_action(
            state,
            player,
            pending_op,
            pending_arg,
            quantity,
            hire_mult=hire_mult,
            shed_capacity=shed_capacity,
        )
        next_slot, next_phase = _advance_market(market_slot)
        return new_state, new_action, next_phase, unit, next_slot, pending_op, pending_arg

    def done(_):
        return state, action, phase, unit, market_slot, pending_op, pending_arg

    return jax.lax.switch(
        phase, (unit_op, unit_quantity, market_op, market_quantity, done), operand=None
    )


def sample_actions(
    model: M.PolicyValueNet,
    variables: dict,
    cache_template: dict,
    states: State,
    players: jnp.ndarray,
    key: jax.Array,
    counters: T.EpisodeCounters | None = None,
    temperature: float = 1.0,
    greedy: bool = False,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1.0,
) -> PolicyOutput:
    """batched Stateから複合行動をGPU上で自己回帰生成する。"""
    batch_size = players.shape[0]
    if counters is None:
        zero = jnp.zeros((batch_size, C.N_PRODUCTS), dtype=jnp.float32)
        counters = T.EpisodeCounters(zero, zero, zero, zero, zero.astype(bool))
    encoder = jax.vmap(
        lambda state, player, counter: T.encode_observation(state, player, counter, turns_per_day)
    )(states, players, counters)
    memory = model.apply(variables, encoder.index, encoder.value, method=model.encode)
    privileged = None
    if model.config.use_asymmetric_critic:
        privileged_features, privileged_positions, privileged_padding = jax.vmap(
            T.encode_privileged
        )(states, players)
        privileged = model.apply(
            variables,
            privileged_features.index,
            privileged_features.value,
            privileged_positions,
            privileged_padding,
            method=model.encode_privileged,
        )
    policy_value = model.apply(variables, memory, privileged, method=model.get_value)

    keys = jax.random.split(key, batch_size)
    carry = _DecodeCarry(
        shadow=states,
        action=_empty_action(batch_size),
        phase=jnp.full((batch_size,), _UNIT_OP, dtype=jnp.int32),
        unit=jnp.zeros((batch_size,), dtype=jnp.int32),
        n_units=1 + jnp.sum(states.hands_active, axis=-1)[jnp.arange(batch_size), players],
        market_slot=jnp.zeros((batch_size,), dtype=jnp.int32),
        pending_op=jnp.zeros((batch_size,), dtype=jnp.int32),
        pending_arg=jnp.zeros((batch_size,), dtype=jnp.int32),
        previous_index=jnp.zeros((batch_size, F.MAX_CANDIDATE_FEATURES), dtype=jnp.int32),
        previous_value=jnp.zeros((batch_size, F.MAX_CANDIDATE_FEATURES), dtype=jnp.float32),
        log_prob=jnp.zeros((batch_size,), dtype=jnp.float32),
        entropy=jnp.zeros((batch_size,), dtype=jnp.float32),
        num_decisions=jnp.zeros((batch_size,), dtype=jnp.int32),
        keys=keys,
        cache=jax.tree.map(jnp.zeros_like, cache_template),
    )

    def step(current: _DecodeCarry, _):
        token_index, token_value, position, board_position = _decoder_token(current, players)
        decoder_variables = freeze({"params": variables["params"], "cache": current.cache})
        hidden, mutable = model.apply(
            decoder_variables,
            memory,
            token_index,
            token_value,
            position,
            board_position,
            method=model.decode_step,
            mutable=["cache"],
        )
        candidate_index, candidate_value, candidate_mask = _candidate_batch(
            current, players, turns_per_day, shed_capacity, hire_mult
        )
        scores = model.apply(
            variables,
            hidden[:, 0],
            candidate_index,
            candidate_value,
            candidate_mask,
            method=model.score_candidates,
        )
        scores = scores / jnp.asarray(temperature, dtype=scores.dtype)
        log_probs = jax.nn.log_softmax(scores)
        probabilities = jnp.exp(log_probs)
        safe_log_probs = jnp.where(candidate_mask, log_probs, 0.0)
        row_entropy = -jnp.sum(probabilities * safe_log_probs, axis=-1)
        split = jax.vmap(lambda rng: jax.random.split(rng, 2))(current.keys)
        next_keys, sample_keys = split[:, 0], split[:, 1]
        sampled = jax.vmap(jax.random.categorical)(sample_keys, scores)
        choice = jnp.where(greedy, jnp.argmax(scores, axis=-1), sampled)
        active = current.phase != _DONE
        chosen_log_prob = jnp.take_along_axis(log_probs, choice[:, None], axis=1)[:, 0]
        batch_rows = jnp.arange(batch_size)
        previous_index = candidate_index[batch_rows, choice]
        previous_value = candidate_value[batch_rows, choice]
        transitioned = jax.vmap(
            _transition_one,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None),
        )(
            current.shadow,
            current.action,
            current.phase,
            current.unit,
            current.n_units,
            current.market_slot,
            current.pending_op,
            current.pending_arg,
            choice,
            players,
            turns_per_day,
            shed_capacity,
            hire_mult,
        )
        next_carry = _DecodeCarry(
            shadow=transitioned[0],
            action=transitioned[1],
            phase=transitioned[2],
            unit=transitioned[3],
            n_units=current.n_units,
            market_slot=transitioned[4],
            pending_op=transitioned[5],
            pending_arg=transitioned[6],
            previous_index=jnp.where(active[:, None], previous_index, current.previous_index),
            previous_value=jnp.where(active[:, None], previous_value, current.previous_value),
            log_prob=current.log_prob + jnp.where(active, chosen_log_prob, 0.0),
            entropy=current.entropy + jnp.where(active, row_entropy, 0.0),
            num_decisions=current.num_decisions + active.astype(jnp.int32),
            keys=next_keys,
            cache=mutable["cache"],
        )
        return next_carry, (
            choice,
            active,
            jnp.where(active, chosen_log_prob, 0.0),
            jnp.where(active, row_entropy, 0.0),
        )

    carry, trace = jax.lax.scan(step, carry, xs=None, length=L.MAX_DECODE_LEN)
    choices, decision_mask, token_log_prob, token_entropy = (
        jnp.swapaxes(value, 0, 1) for value in trace
    )
    return PolicyOutput(
        action=carry.action,
        value=policy_value,
        log_prob=jnp.sum(token_log_prob, axis=-1),
        entropy=jnp.sum(token_entropy, axis=-1),
        num_decisions=carry.num_decisions,
        choices=choices,
        decision_mask=decision_mask,
        token_log_prob=token_log_prob,
        token_entropy=token_entropy,
    )


def sample_self_play_actions(
    model: M.PolicyValueNet,
    variables: dict,
    cache_template: dict,
    states: State,
    key: jax.Array,
    counters: T.EpisodeCounters | None = None,
    temperature: float = 1.0,
    greedy: bool = False,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1.0,
) -> SelfPlayOutput:
    """各局の両プレイヤーを1バッチに束ねて複合行動を生成する。"""
    batch_size = states.step.shape[0]
    doubled_states = jax.tree.map(lambda value: jnp.concatenate([value, value]), states)
    players = jnp.concatenate(
        [jnp.zeros((batch_size,), dtype=jnp.int32), jnp.ones((batch_size,), dtype=jnp.int32)]
    )
    doubled_counters = None
    if counters is not None:
        doubled_counters = jax.tree.map(
            lambda value: jnp.concatenate([value[:, 0], value[:, 1]]), counters
        )
    output = sample_actions(
        model,
        variables,
        cache_template,
        doubled_states,
        players,
        key,
        counters=doubled_counters,
        temperature=temperature,
        greedy=greedy,
        turns_per_day=turns_per_day,
        shed_capacity=shed_capacity,
        hire_mult=hire_mult,
    )
    player0 = jax.tree.map(lambda value: value[:batch_size], output.action)
    player1 = jax.tree.map(lambda value: value[batch_size:], output.action)

    def pair(value):
        return jnp.stack([value[:batch_size], value[batch_size:]], axis=1)

    return SelfPlayOutput(
        action=combine_player_actions(player0, player1),
        value=pair(output.value),
        log_prob=pair(output.log_prob),
        entropy=pair(output.entropy),
        num_decisions=pair(output.num_decisions),
        choices=pair(output.choices),
        decision_mask=pair(output.decision_mask),
        token_log_prob=pair(output.token_log_prob),
        token_entropy=pair(output.token_entropy),
    )


def evaluate_choices(
    model: M.PolicyValueNet,
    variables: dict,
    states: State,
    players: jnp.ndarray,
    choices: jnp.ndarray,
    counters: T.EpisodeCounters | None = None,
    temperature: float = 1.0,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1.0,
) -> EvaluationOutput:
    """rollout時の候補index列を現在の方策で教師強制評価する。"""
    batch_size = players.shape[0]
    if counters is None:
        zero = jnp.zeros((batch_size, C.N_PRODUCTS), dtype=jnp.float32)
        counters = T.EpisodeCounters(zero, zero, zero, zero, zero.astype(bool))
    encoder = jax.vmap(
        lambda state, player, counter: T.encode_observation(state, player, counter, turns_per_day)
    )(states, players, counters)
    memory = model.apply(variables, encoder.index, encoder.value, method=model.encode)

    initial = _DecodeCarry(
        shadow=states,
        action=_empty_action(batch_size),
        phase=jnp.full((batch_size,), _UNIT_OP, dtype=jnp.int32),
        unit=jnp.zeros((batch_size,), dtype=jnp.int32),
        n_units=1 + jnp.sum(states.hands_active, axis=-1)[jnp.arange(batch_size), players],
        market_slot=jnp.zeros((batch_size,), dtype=jnp.int32),
        pending_op=jnp.zeros((batch_size,), dtype=jnp.int32),
        pending_arg=jnp.zeros((batch_size,), dtype=jnp.int32),
        previous_index=jnp.zeros((batch_size, F.MAX_CANDIDATE_FEATURES), dtype=jnp.int32),
        previous_value=jnp.zeros((batch_size, F.MAX_CANDIDATE_FEATURES), dtype=jnp.float32),
        log_prob=jnp.zeros((batch_size,), dtype=jnp.float32),
        entropy=jnp.zeros((batch_size,), dtype=jnp.float32),
        num_decisions=jnp.zeros((batch_size,), dtype=jnp.int32),
        keys=jnp.zeros((batch_size, 2), dtype=jnp.uint32),
        cache={},
    )

    def trace_step(current: _DecodeCarry, choice: jnp.ndarray):
        token_index, token_value, position, board_position = _decoder_token(current, players)
        candidate_index, candidate_value, candidate_mask = _candidate_batch(
            current, players, turns_per_day, shed_capacity, hire_mult
        )
        active = current.phase != _DONE
        rows = jnp.arange(batch_size)
        previous_index = candidate_index[rows, choice]
        previous_value = candidate_value[rows, choice]
        phase = current.phase
        transitioned = jax.vmap(
            _transition_one,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None),
        )(
            current.shadow,
            current.action,
            current.phase,
            current.unit,
            current.n_units,
            current.market_slot,
            current.pending_op,
            current.pending_arg,
            choice,
            players,
            turns_per_day,
            shed_capacity,
            hire_mult,
        )
        next_carry = current._replace(
            shadow=transitioned[0],
            action=transitioned[1],
            phase=transitioned[2],
            unit=transitioned[3],
            market_slot=transitioned[4],
            pending_op=transitioned[5],
            pending_arg=transitioned[6],
            previous_index=jnp.where(active[:, None], previous_index, current.previous_index),
            previous_value=jnp.where(active[:, None], previous_value, current.previous_value),
            num_decisions=current.num_decisions + active.astype(jnp.int32),
        )
        trace = (
            token_index[:, 0],
            token_value[:, 0],
            position[:, 0],
            board_position[:, 0],
            candidate_mask,
            phase,
            active,
        )
        return next_carry, trace

    final, trace = jax.lax.scan(trace_step, initial, jnp.swapaxes(choices, 0, 1))
    token_index, token_value, positions, board_positions, masks, phases, active = (
        jnp.swapaxes(item, 0, 1) for item in trace
    )
    hidden = model.apply(
        variables,
        memory,
        token_index,
        token_value,
        positions,
        board_positions,
        ~active,
        method=model.decode,
    )
    unit_phase = phases == _UNIT_OP
    market_phase = phases == _MARKET_OP
    candidate_index = jnp.where(
        unit_phase[..., None, None],
        _UNIT_INDEX[None, None],
        jnp.where(
            market_phase[..., None, None],
            _MARKET_INDEX[None, None],
            A.QUANTITY_INDEX[None, None],
        ),
    )
    candidate_value = jnp.where(
        unit_phase[..., None, None],
        _UNIT_VALUE[None, None],
        jnp.where(
            market_phase[..., None, None],
            _MARKET_VALUE[None, None],
            A.QUANTITY_VALUE[None, None],
        ),
    )
    scores = model.apply(
        variables,
        hidden,
        candidate_index,
        candidate_value,
        masks,
        method=model.score_candidates,
    )
    scores = scores / jnp.asarray(temperature, dtype=scores.dtype)
    log_probs = jax.nn.log_softmax(scores)
    selected = jnp.take_along_axis(log_probs, choices[..., None], axis=-1)[..., 0]
    probabilities = jnp.exp(log_probs)
    entropy = -jnp.sum(probabilities * jnp.where(masks, log_probs, 0.0), axis=-1)

    privileged = None
    if model.config.use_asymmetric_critic:
        privileged_features, privileged_positions, privileged_padding = jax.vmap(
            T.encode_privileged
        )(states, players)
        privileged = model.apply(
            variables,
            privileged_features.index,
            privileged_features.value,
            privileged_positions,
            privileged_padding,
            method=model.encode_privileged,
        )
    value = model.apply(variables, memory, privileged, method=model.get_value)
    return EvaluationOutput(
        value=value,
        log_prob=jnp.sum(jnp.where(active, selected, 0.0), axis=-1),
        entropy=jnp.sum(jnp.where(active, entropy, 0.0), axis=-1),
        num_decisions=final.num_decisions,
        token_log_prob=jnp.where(active, selected, 0.0),
        token_entropy=jnp.where(active, entropy, 0.0),
    )
