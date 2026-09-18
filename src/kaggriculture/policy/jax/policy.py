"""非自己回帰方策の一括サンプリングとPPO再評価。"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.jax import actions as A
from kaggriculture.policy.jax import executor as E
from kaggriculture.policy.jax import model as M
from kaggriculture.policy.jax import strategy as S
from kaggriculture.policy.jax import tokenize as T
from kaggriculture.policy.jax.types import EvaluationOutput, Intent, PolicyOutput
from kaggriculture.rules import constants as C
from kaggriculture.simulator.action import Action
from kaggriculture.simulator.state import State


def _inputs(states: State, players: jnp.ndarray, counters, turns_per_day: int):
    batch = players.shape[0]
    if counters is None:
        zero = jnp.zeros((batch, C.N_PRODUCTS), dtype=jnp.float32)
        counters = T.EpisodeCounters(zero, zero, zero, zero, zero.astype(bool))
    encoded = jax.vmap(
        lambda state, player, counter: T.encode_observation(state, player, counter, turns_per_day)
    )(states, players, counters)
    unit_inventory = jax.vmap(T.encode_unit_inventories)(states, players)
    positions_xy = jnp.concatenate(
        [
            states.farmer_pos[jnp.arange(batch), players, None],
            states.hands_pos[jnp.arange(batch), players],
        ],
        axis=1,
    )
    board_size = states.tiles_kind.shape[-1]
    positions = positions_xy[..., 1] * board_size + positions_xy[..., 0]
    active = jnp.concatenate(
        [jnp.ones((batch, 1), dtype=bool), states.hands_active[jnp.arange(batch), players]],
        axis=1,
    )
    positions = jnp.where(active, positions, L.NO_POSITION)
    return encoded, positions, active, counters, unit_inventory


def _candidate_logits(model, variables, hidden, index, value, mask):
    return model.apply(variables, hidden, index, value, mask, method=model.score_candidates)


def _distribution(logits, choices=None, key=None, greedy=False):
    log_probs = jax.nn.log_softmax(logits)
    probabilities = jnp.exp(log_probs)
    entropy = -jnp.sum(jnp.where(jnp.isfinite(log_probs), probabilities * log_probs, 0.0), axis=-1)
    if choices is None:
        sampled = jax.random.categorical(key, logits, axis=-1)
        choices = jnp.where(greedy, jnp.argmax(logits, axis=-1), sampled)
    selected = jnp.take_along_axis(log_probs, choices[..., None], axis=-1)[..., 0]
    return choices, selected, entropy


def _logits(
    model: M.PolicyValueNet,
    variables: dict,
    states: State,
    players: jnp.ndarray,
    counters,
    turns_per_day: int,
    shed_capacity: int,
):
    encoded, positions, unit_active, _, unit_inventory = _inputs(
        states, players, counters, turns_per_day
    )
    privileged = {}
    if model.config.use_asymmetric_critic:
        features, privileged_positions, privileged_padding = jax.vmap(T.encode_privileged)(
            states, players
        )
        privileged = {
            "privileged_index": features.index,
            "privileged_value": features.value,
            "privileged_positions": privileged_positions,
            "privileged_padding": privileged_padding,
        }
    queries, value = model.apply(
        variables,
        encoded.index,
        encoded.value,
        positions,
        unit_active,
        unit_inventory.index,
        unit_inventory.value,
        deterministic=True,
        **privileged,
    )
    unit_hidden = queries[:, : M.N_UNIT_SLOTS]
    market_hidden = queries[:, M.N_UNIT_SLOTS :]
    units = jnp.arange(M.N_UNIT_SLOTS)
    unit_mask = jax.vmap(
        lambda state, player: jax.vmap(
            lambda unit: (
                A.legal_unit_mask(
                    state, player, unit, turns_per_day=turns_per_day, shed_capacity=shed_capacity
                )
                & S.unit_mask(state, player, unit)
            )
        )(units)
    )(states, players)
    unit_mask = unit_mask & unit_active[..., None]
    unit_mask = unit_mask.at[..., 0].set(True)
    unit_logits = _candidate_logits(
        model,
        variables,
        unit_hidden,
        A.UNIT_CANDIDATES.index[None, None],
        A.UNIT_CANDIDATES.value[None, None],
        unit_mask,
    )
    # 先行注文で合法化し得る候補は残し、状態に依存しない戦略規則だけ適用する。
    market_mask = jax.vmap(S.market_mask)(states, players)
    market_logits = _candidate_logits(
        model,
        variables,
        market_hidden,
        A.MARKET_CANDIDATES.index[None, None],
        A.MARKET_CANDIDATES.value[None, None],
        market_mask,
    )
    return value, unit_hidden, market_hidden, unit_logits, market_logits, unit_active, unit_mask


def _market_active(choices: jnp.ndarray) -> jnp.ndarray:
    """最初のSTOPまで（STOP自身を含む）の有効slot maskを返す。"""
    is_stop = A.MARKET_CANDIDATES.op[choices] == A.MARKET_STOP
    stopped_before = jnp.cumsum(is_stop.astype(jnp.int32), axis=-1) - is_stop.astype(jnp.int32)
    return stopped_before == 0


def _quantity_logits(model, variables, hidden, selected, table, mask):
    selected_index = table.index[selected]
    selected_value = table.value[selected]
    conditioned = model.apply(
        variables,
        hidden,
        selected_index,
        selected_value,
        method=model.condition_quantity,
    )
    return _candidate_logits(
        model,
        variables,
        conditioned,
        A.QUANTITY_INDEX[None, None],
        A.QUANTITY_VALUE[None, None],
        mask,
    )


def sample_actions(
    model: M.PolicyValueNet,
    variables: dict,
    states: State,
    players: jnp.ndarray,
    key: jax.Array,
    counters=None,
    temperature: float = 1.0,
    greedy: bool = False,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1.0,
) -> PolicyOutput:
    """全slotを1回のTransformer計算で共同生成する。"""
    value, unit_hidden, market_hidden, unit_logits, market_logits, unit_active, _ = _logits(
        model, variables, states, players, counters, turns_per_day, shed_capacity
    )
    keys = jax.random.split(key, 4)
    unit, unit_lp, unit_entropy = _distribution(
        unit_logits / temperature, key=keys[0], greedy=greedy
    )
    market, market_lp, market_entropy = _distribution(
        market_logits / temperature, key=keys[1], greedy=greedy
    )
    batch = players.shape[0]
    units = jnp.arange(M.N_UNIT_SLOTS)
    unit_op = A.UNIT_CANDIDATES.op[unit]
    unit_arg = A.UNIT_CANDIDATES.arg[unit]
    unit_max = jax.vmap(
        lambda state, player, ops, args: jax.vmap(
            lambda index, op, arg: A.unit_quantity_upper_bound(state, player, index, op, arg)
        )(units, ops, args)
    )(states, players, unit_op, unit_arg)
    unit_needs = jax.vmap(
        lambda state, player, ops, args: jax.vmap(
            lambda index, op, arg: A.unit_requires_quantity(state, player, index, op, arg)
        )(units, ops, args)
    )(states, players, unit_op, unit_arg)
    numbers = jnp.arange(1, A.QUANTITY_INDEX.shape[0] + 1)
    unit_qmask = numbers[None, None] <= jnp.maximum(unit_max[..., None], 1)
    unit_qlogits = _quantity_logits(
        model, variables, unit_hidden, unit, A.UNIT_CANDIDATES, unit_qmask
    )
    unit_quantity, unit_qlp, unit_qentropy = _distribution(
        unit_qlogits / temperature, key=keys[2], greedy=greedy
    )

    market_op = A.MARKET_CANDIDATES.op[market]
    market_active = _market_active(market)
    market_needs = market_active & (
        (market_op == C.MARKET_OP_BUY_SEED)
        | (market_op == C.MARKET_OP_BUY_PRODUCT)
        | (market_op == C.MARKET_OP_BUY_ANIMAL)
        | (market_op == C.MARKET_OP_SELL)
    )
    # slot間依存はExecutorが扱うため、数量headは1..100を区別して全て出力する。
    market_qmask = jnp.ones((batch, C.MAX_MARKET_ORDERS, A.QUANTITY_INDEX.shape[0]), dtype=bool)
    market_qlogits = _quantity_logits(
        model, variables, market_hidden, market, A.MARKET_CANDIDATES, market_qmask
    )
    market_quantity, market_qlp, market_qentropy = _distribution(
        market_qlogits / temperature, key=keys[3], greedy=greedy
    )
    intent = Intent(unit, unit_quantity, market, market_quantity)
    action, stats = E.execute(
        states,
        players,
        intent,
        turns_per_day=turns_per_day,
        shed_capacity=shed_capacity,
        hire_mult=hire_mult,
    )
    unit_valid = unit_active
    slot_log_prob = jnp.concatenate(
        [
            jnp.where(unit_valid, unit_lp + jnp.where(unit_needs, unit_qlp, 0.0), 0.0),
            jnp.where(market_active, market_lp + jnp.where(market_needs, market_qlp, 0.0), 0.0),
        ],
        axis=1,
    )
    entropy = jnp.concatenate(
        [
            jnp.where(unit_valid, unit_entropy + jnp.where(unit_needs, unit_qentropy, 0.0), 0.0),
            jnp.where(
                market_active,
                market_entropy + jnp.where(market_needs, market_qentropy, 0.0),
                0.0,
            ),
        ],
        axis=1,
    )
    slot_mask = jnp.concatenate([unit_valid, market_active], axis=1)
    return PolicyOutput(
        intent, action, slot_log_prob.sum(-1), slot_log_prob, slot_mask, entropy, value, stats
    )


def evaluate_intent(
    model: M.PolicyValueNet,
    variables: dict,
    states: State,
    players: jnp.ndarray,
    intent: Intent,
    counters=None,
    temperature: float = 1.0,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
) -> EvaluationOutput:
    """保存したintentを一括再評価する。PPO用の逐次traceは不要。"""
    value, uh, mh, ul, ml, active, unit_mask = _logits(
        model, variables, states, players, counters, turns_per_day, shed_capacity
    )
    _, ulp, ue = _distribution(ul / temperature, choices=intent.unit)
    _, mlp, me = _distribution(ml / temperature, choices=intent.market)
    uop = A.UNIT_CANDIDATES.op[intent.unit]
    uarg = A.UNIT_CANDIDATES.arg[intent.unit]
    units = jnp.arange(M.N_UNIT_SLOTS)
    uneeds = jax.vmap(
        lambda state, player, ops, args: jax.vmap(
            lambda index, op, arg: A.unit_requires_quantity(state, player, index, op, arg)
        )(units, ops, args)
    )(states, players, uop, uarg)
    umax = jax.vmap(
        lambda state, player, ops, args: jax.vmap(
            lambda index, op, arg: A.unit_quantity_upper_bound(state, player, index, op, arg)
        )(units, ops, args)
    )(states, players, uop, uarg)
    numbers = jnp.arange(1, A.QUANTITY_INDEX.shape[0] + 1)
    uql = _quantity_logits(
        model,
        variables,
        uh,
        intent.unit,
        A.UNIT_CANDIDATES,
        numbers[None, None] <= jnp.maximum(umax[..., None], 1),
    )
    _, uqlp, uqe = _distribution(uql / temperature, choices=intent.unit_quantity)
    unit_candidate_valid = jnp.take_along_axis(unit_mask, intent.unit[..., None], axis=-1)[..., 0]
    unit_quantity_mask = numbers[None, None] <= jnp.maximum(umax[..., None], 1)
    unit_quantity_valid = jnp.take_along_axis(
        unit_quantity_mask, intent.unit_quantity[..., None], axis=-1
    )[..., 0]
    unit_valid = unit_candidate_valid & (~uneeds | unit_quantity_valid)
    mop = A.MARKET_CANDIDATES.op[intent.market]
    mactive = _market_active(intent.market)
    mneeds = mactive & (
        (mop == C.MARKET_OP_BUY_SEED)
        | (mop == C.MARKET_OP_BUY_PRODUCT)
        | (mop == C.MARKET_OP_BUY_ANIMAL)
        | (mop == C.MARKET_OP_SELL)
    )
    mqmask = jnp.ones((*intent.market.shape, A.QUANTITY_INDEX.shape[0]), dtype=bool)
    mql = _quantity_logits(model, variables, mh, intent.market, A.MARKET_CANDIDATES, mqmask)
    _, mqlp, mqe = _distribution(mql / temperature, choices=intent.market_quantity)
    slot_lp = jnp.concatenate(
        [
            jnp.where(active, ulp + jnp.where(uneeds, uqlp, 0), 0),
            jnp.where(mactive, mlp + jnp.where(mneeds, mqlp, 0), 0),
        ],
        axis=1,
    )
    entropy = jnp.concatenate(
        [
            jnp.where(active, ue + jnp.where(uneeds, uqe, 0), 0),
            jnp.where(mactive, me + jnp.where(mneeds, mqe, 0), 0),
        ],
        axis=1,
    )
    slot_mask = jnp.concatenate([active, mactive], axis=1)
    slot_valid = jnp.concatenate([unit_valid, jnp.ones_like(mactive)], axis=1)
    return EvaluationOutput(slot_lp.sum(-1), slot_lp, slot_mask, slot_valid, entropy, value)


def sample_self_play_actions(model, variables, states, key, counters=None, **kwargs):
    """両プレイヤーを同じ大batchとして一括生成する。"""
    batch = states.step.shape[0]
    doubled = jax.tree.map(lambda x: jnp.concatenate([x, x]), states)
    players = jnp.concatenate([jnp.zeros(batch, jnp.int32), jnp.ones(batch, jnp.int32)])
    doubled_counters = None
    if counters is not None:
        doubled_counters = jax.tree.map(lambda x: jnp.concatenate([x[:, 0], x[:, 1]]), counters)
    output = sample_actions(model, variables, doubled, players, key, doubled_counters, **kwargs)
    fields = [jnp.stack([x[:batch], x[batch:]], axis=1) for x in output.action]
    action = Action(*fields)

    def reshape(x):
        return jnp.stack([x[:batch], x[batch:]], axis=1)

    return output._replace(
        action=action,
        intent=jax.tree.map(reshape, output.intent),
        log_prob=reshape(output.log_prob),
        slot_log_prob=reshape(output.slot_log_prob),
        slot_mask=reshape(output.slot_mask),
        entropy=reshape(output.entropy),
        value=reshape(output.value),
        stats=jax.tree.map(reshape, output.stats),
    )


def state_values(model, variables, states, counters=None, turns_per_day=24):
    """batch内の両プレイヤー価値を返す。"""
    batch = states.step.shape[0]
    doubled = jax.tree.map(lambda x: jnp.concatenate([x, x]), states)
    players = jnp.concatenate([jnp.zeros(batch, jnp.int32), jnp.ones(batch, jnp.int32)])
    doubled_counters = None
    if counters is not None:
        doubled_counters = jax.tree.map(lambda x: jnp.concatenate([x[:, 0], x[:, 1]]), counters)
    encoded, positions, active, _, unit_inventory = _inputs(
        doubled, players, doubled_counters, turns_per_day
    )
    privileged = {}
    if model.config.use_asymmetric_critic:
        features, privileged_positions, privileged_padding = jax.vmap(T.encode_privileged)(
            doubled, players
        )
        privileged = {
            "privileged_index": features.index,
            "privileged_value": features.value,
            "privileged_positions": privileged_positions,
            "privileged_padding": privileged_padding,
        }
    _, values = model.apply(
        variables,
        encoded.index,
        encoded.value,
        positions,
        active,
        unit_inventory.index,
        unit_inventory.value,
        deterministic=True,
        **privileged,
    )
    return jnp.stack([values[:batch], values[batch:]], axis=1)


def initialize(model: M.PolicyValueNet, key, batch_size: int = 1):
    """全parameterを固定shape入力で初期化する。"""
    from kaggriculture.policy.jax import features as F

    index = jnp.zeros((batch_size, L.NUM_WORDS_ENCODER, F.MAX_ENCODER_FEATURES), jnp.int32)
    value = jnp.zeros_like(index, dtype=jnp.float32)
    positions = jnp.full((batch_size, M.N_UNIT_SLOTS), L.NO_POSITION, jnp.int32)
    active = jnp.zeros((batch_size, M.N_UNIT_SLOTS), bool).at[:, 0].set(True)
    unit_inventory_index = jnp.zeros(
        (batch_size, M.N_UNIT_SLOTS, F.MAX_ENCODER_FEATURES), jnp.int32
    )
    unit_inventory_value = jnp.zeros_like(unit_inventory_index, dtype=jnp.float32)
    privileged = {}
    if model.config.use_asymmetric_critic:
        from kaggriculture.policy.jax import features as F

        count = len(L.PRIVILEGED_OWNER_ZONE_WITH_CLS) - 1
        privileged = {
            "privileged_index": jnp.zeros(
                (batch_size, count, F.MAX_PRIVILEGED_FEATURES), jnp.int32
            ),
            "privileged_value": jnp.zeros(
                (batch_size, count, F.MAX_PRIVILEGED_FEATURES), jnp.float32
            ),
            "privileged_positions": jnp.full((batch_size, count), L.NO_POSITION, jnp.int32),
            "privileged_padding": jnp.zeros((batch_size, count), bool),
        }
    return model.init(
        key,
        index,
        value,
        positions,
        active,
        unit_inventory_index,
        unit_inventory_value,
        deterministic=True,
        **privileged,
    )
