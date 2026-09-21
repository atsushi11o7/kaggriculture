"""固定slot非自己回帰方策のPPO目的関数。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import traverse_util
from flax.training.train_state import TrainState

from kaggriculture.policy.common.config import CRITIC_PARAMETER_MODULES
from kaggriculture.policy.jax import model as M
from kaggriculture.policy.jax import policy as P
from kaggriculture.policy.jax.tokenize import EpisodeCounters
from kaggriculture.policy.jax.types import Intent
from kaggriculture.simulator.state import State
from kaggriculture.training.rl import compute_gae, terminal_win_rewards


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    reference_actor_l2_coef: float = 0.0
    target_kl: float | None = 0.02
    max_grad_norm: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    temperature: float = 0.8
    turns_per_day: int = 24
    shed_capacity: int = 100


class PPOBatch(NamedTuple):
    states: State
    players: jnp.ndarray
    intent: Intent
    slot_mask: jnp.ndarray
    old_slot_log_prob: jnp.ndarray
    old_value: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    counters: EpisodeCounters | None


class CriticBatch(NamedTuple):
    states: State
    targets: jnp.ndarray
    mask: jnp.ndarray
    counters: EpisodeCounters | None


class CriticMetrics(NamedTuple):
    loss: jnp.ndarray
    r2: jnp.ndarray
    correlation: jnp.ndarray


class Metrics(NamedTuple):
    loss: jnp.ndarray
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    entropy: jnp.ndarray
    approx_kl: jnp.ndarray
    clip_fraction: jnp.ndarray
    reference_actor_l2: jnp.ndarray


def create_critic_train_state(
    model: M.PolicyValueNet,
    variables: dict,
    *,
    learning_rate: float,
    max_grad_norm: float,
):
    """Actorを固定し、critic moduleだけを更新するTrainStateを作る。"""
    labels = jax.tree_util.tree_map_with_path(
        lambda path, _: "critic" if path[0].key in CRITIC_PARAMETER_MODULES else "actor",
        variables["params"],
    )
    optimizer = optax.multi_transform(
        {
            "critic": optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate),
            ),
            "actor": optax.set_to_zero(),
        },
        labels,
    )
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optimizer)


def create_train_state(model: M.PolicyValueNet, variables: dict, config: PPOConfig):
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
    )
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optimizer)


def _critic_predictions(model, params, batch: CriticBatch, turns_per_day: int):
    return P.state_values(
        model,
        {"params": params},
        batch.states,
        counters=batch.counters,
        turns_per_day=turns_per_day,
    )


def _critic_loss(model, params, batch: CriticBatch, turns_per_day: int):
    predictions = _critic_predictions(model, params, batch, turns_per_day)
    mask = batch.mask.astype(jnp.float32)
    count = jnp.maximum(mask.sum(), 1)
    return jnp.sum(jnp.square(predictions - batch.targets) * mask) / count


@partial(jax.jit, static_argnums=(0, 3))
def update_critic_minibatch(model, train_state, batch: CriticBatch, turns_per_day: int):
    loss, gradients = jax.value_and_grad(_critic_loss, argnums=1)(
        model, train_state.params, batch, turns_per_day
    )
    return train_state.apply_gradients(grads=gradients), loss


@partial(jax.jit, static_argnums=(0, 4, 5, 6))
def update_critic_epochs(
    model,
    train_state,
    batch: CriticBatch,
    key,
    num_epochs: int,
    minibatch_size: int,
    turns_per_day: int,
):
    """固定したMonte Carlo教師でcriticだけを複数epoch更新する。"""
    sample_count = batch.targets.shape[0]
    if sample_count % minibatch_size:
        raise ValueError("critic sample count must be divisible by minibatch_size")
    minibatches = sample_count // minibatch_size

    def epoch_step(carry, _):
        state, rng = carry
        rng, permutation_key = jax.random.split(rng)
        indices = jax.random.permutation(permutation_key, sample_count).reshape(
            minibatches, minibatch_size
        )

        def minibatch_step(current, selected):
            minibatch = jax.tree.map(
                lambda value: value[selected] if value is not None else None,
                batch,
                is_leaf=lambda value: value is None,
            )
            return update_critic_minibatch(model, current, minibatch, turns_per_day)

        state, losses = jax.lax.scan(minibatch_step, state, indices)
        return (state, rng), jnp.mean(losses)

    (train_state, _), losses = jax.lax.scan(
        epoch_step, (train_state, key), xs=None, length=num_epochs
    )
    return train_state, jnp.mean(losses)


@partial(jax.jit, static_argnums=(0, 3, 4))
def evaluate_critic(
    model, params, batch: CriticBatch, turns_per_day: int, minibatch_size: int
) -> CriticMetrics:
    """masked Monte Carlo教師に対するMSE・R²・相関を返す。

    全局面を一度に順伝播するとメモリを超えるため、minibatch単位で予測する。
    """
    sample_count = batch.targets.shape[0]
    if sample_count % minibatch_size:
        raise ValueError("critic sample count must be divisible by minibatch_size")
    chunks = sample_count // minibatch_size
    chunked = jax.tree.map(
        lambda value: value.reshape((chunks, minibatch_size) + value.shape[1:]),
        (batch.states, batch.counters),
    )

    def predict(item):
        states, counters = item
        return _critic_predictions(
            model, params, CriticBatch(states, None, None, counters), turns_per_day
        )

    predictions = jax.lax.map(predict, chunked).reshape(batch.targets.shape)
    mask = batch.mask.astype(jnp.float32)
    count = jnp.maximum(mask.sum(), 1)
    target_mean = jnp.sum(batch.targets * mask) / count
    prediction_mean = jnp.sum(predictions * mask) / count
    target_delta = batch.targets - target_mean
    prediction_delta = predictions - prediction_mean
    residual = predictions - batch.targets
    sse = jnp.sum(jnp.square(residual) * mask)
    target_ss = jnp.sum(jnp.square(target_delta) * mask)
    prediction_ss = jnp.sum(jnp.square(prediction_delta) * mask)
    covariance = jnp.sum(target_delta * prediction_delta * mask)
    loss = sse / count
    r2 = 1 - sse / jnp.maximum(target_ss, 1e-8)
    correlation = covariance / jnp.sqrt(jnp.maximum(target_ss * prediction_ss, 1e-8))
    return CriticMetrics(loss, r2, correlation)


def _normalize_advantages(advantages: jnp.ndarray, sample_mask: jnp.ndarray) -> jnp.ndarray:
    """固定相手を除いたlearner行だけでadvantageを標準化する。"""
    sample_mask = sample_mask.astype(jnp.float32)
    sample_count = jnp.maximum(sample_mask.sum(), 1)
    mean = jnp.sum(advantages * sample_mask) / sample_count
    variance = jnp.sum(jnp.square(advantages - mean) * sample_mask) / sample_count
    return (advantages - mean) / jnp.sqrt(variance + 1e-8)


def _reference_actor_l2(params, reference_params) -> jnp.ndarray:
    """参照Actorと共有encoderの相対パラメータ距離を返す。"""
    current = traverse_util.flatten_dict(params)
    reference = traverse_util.flatten_dict(reference_params)
    total = jnp.asarray(0.0)
    count = 0
    for path, value in current.items():
        if path[0] in CRITIC_PARAMETER_MODULES:
            continue
        anchor = reference[path]
        scale = jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(anchor))), 0.1)
        total = total + jnp.sum(jnp.square((value - anchor) / scale))
        count += value.size
    return total / max(count, 1)


def _loss(model, params, batch: PPOBatch, config: PPOConfig, reference_params=None):
    evaluation = P.evaluate_intent(
        model,
        {"params": params},
        batch.states,
        batch.players,
        batch.intent,
        counters=batch.counters,
        temperature=config.temperature,
        turns_per_day=config.turns_per_day,
        shed_capacity=config.shed_capacity,
    )
    sample_mask = batch.slot_mask.any(-1).astype(jnp.float32)
    sample_count = jnp.maximum(sample_mask.sum(), 1)
    advantage = _normalize_advantages(batch.advantages, sample_mask)
    log_ratio = evaluation.slot_log_prob - batch.old_slot_log_prob
    ratio = jnp.exp(jnp.clip(log_ratio, -20.0, 20.0))
    unclipped = ratio * advantage[:, None]
    clipped = jnp.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantage[:, None]
    mask = batch.slot_mask.astype(jnp.float32)
    count = jnp.maximum(mask.sum(-1), 1)
    # population対戦(collect_rollout_vs_opponent)では、固定相手側の行は
    # slot_maskが全slotでFalseになる。行内のmasked sumはここまでで既に0に
    # なるが、外側の平均を全サンプル数で取ると相手の行が分母に残ってしまう
    # (policy_loss等は薄まるだけで済むが、value_lossは相手の状態価値まで
    # learnerのcriticに学習させてしまう不整合が起きる)。sample_maskで
    # learner側の行だけを対象に平均する。自己対戦時は全行が有効なので、
    # 従来の挙動と完全に一致する。

    def _sample_mean(per_sample: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(per_sample * sample_mask) / sample_count

    policy_loss = -_sample_mean((jnp.minimum(unclipped, clipped) * mask).sum(-1) / count)
    approx_kl = _sample_mean((((ratio - 1) - log_ratio) * mask).sum(-1) / count)
    clip_fraction = _sample_mean(
        ((jnp.abs(ratio - 1) > config.clip_epsilon) * mask).sum(-1) / count
    )
    delta = evaluation.value - batch.old_value
    clipped_value = batch.old_value + jnp.clip(
        delta, -config.value_clip_epsilon, config.value_clip_epsilon
    )
    value_loss = 0.5 * _sample_mean(
        jnp.maximum(
            jnp.square(evaluation.value - batch.returns),
            jnp.square(clipped_value - batch.returns),
        )
    )
    entropy = _sample_mean((evaluation.entropy * mask).sum(-1) / count)
    reference_actor_l2 = (
        _reference_actor_l2(params, reference_params)
        if config.reference_actor_l2_coef > 0 and reference_params is not None
        else jnp.asarray(0.0)
    )
    loss = (
        policy_loss
        + config.value_coef * value_loss
        - config.entropy_coef * entropy
        + config.reference_actor_l2_coef * reference_actor_l2
    )
    return loss, Metrics(
        loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction, reference_actor_l2
    )


@partial(jax.jit, static_argnums=(0, 3))
def update_minibatch(model, train_state, batch, config, reference_params=None):
    (_, metrics), gradients = jax.value_and_grad(_loss, argnums=1, has_aux=True)(
        model, train_state.params, batch, config, reference_params
    )
    return train_state.apply_gradients(grads=gradients), metrics


@partial(jax.jit, static_argnums=(0, 4, 5, 6))
def update_epochs(
    model, train_state, batch, key, config, num_epochs, minibatch_size, reference_params=None
):
    sample_count = batch.players.shape[0]
    if sample_count % minibatch_size:
        raise ValueError("sample count must be divisible by minibatch_size")
    minibatches = sample_count // minibatch_size

    zero_metrics = Metrics(*(jnp.asarray(0.0) for _ in Metrics._fields))

    def epoch_step(carry, _):
        state, rng, active = carry
        rng, permutation_key = jax.random.split(rng)

        def run_epoch(current):
            indices = jax.random.permutation(permutation_key, sample_count).reshape(
                minibatches, minibatch_size
            )

            def minibatch_step(inner_state, selected):
                minibatch = jax.tree.map(
                    lambda value: value[selected] if value is not None else None,
                    batch,
                    is_leaf=lambda value: value is None,
                )
                return update_minibatch(model, inner_state, minibatch, config, reference_params)

            updated, minibatch_metrics = jax.lax.scan(minibatch_step, current, indices)
            return updated, jax.tree.map(jnp.mean, minibatch_metrics)

        state, metrics = jax.lax.cond(
            active,
            run_epoch,
            lambda current: (current, zero_metrics),
            state,
        )
        executed = active.astype(jnp.float32)
        within_target = (
            jnp.asarray(True) if config.target_kl is None else metrics.approx_kl <= config.target_kl
        )
        return (state, rng, active & within_target), (metrics, executed)

    (train_state, _, _), (metrics, executed) = jax.lax.scan(
        epoch_step, (train_state, key, jnp.asarray(True)), xs=None, length=num_epochs
    )
    epochs_completed = executed.sum().astype(jnp.int32)
    divisor = jnp.maximum(epochs_completed, 1)
    metrics = jax.tree.map(lambda value: value.sum(axis=0) / divisor, metrics)
    return train_state, metrics, epochs_completed


__all__ = ["CriticBatch", "PPOBatch", "PPOConfig", "compute_gae", "terminal_win_rewards"]
