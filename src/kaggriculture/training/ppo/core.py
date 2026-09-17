"""固定slot非自己回帰方策のPPO目的関数。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

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


class Metrics(NamedTuple):
    loss: jnp.ndarray
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    entropy: jnp.ndarray
    approx_kl: jnp.ndarray
    clip_fraction: jnp.ndarray


def create_train_state(model: M.PolicyValueNet, variables: dict, config: PPOConfig):
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
    )
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optimizer)


def _normalize_advantages(advantages: jnp.ndarray, sample_mask: jnp.ndarray) -> jnp.ndarray:
    """固定相手を除いたlearner行だけでadvantageを標準化する。"""
    sample_mask = sample_mask.astype(jnp.float32)
    sample_count = jnp.maximum(sample_mask.sum(), 1)
    mean = jnp.sum(advantages * sample_mask) / sample_count
    variance = jnp.sum(jnp.square(advantages - mean) * sample_mask) / sample_count
    return (advantages - mean) / jnp.sqrt(variance + 1e-8)


def _loss(model, params, batch: PPOBatch, config: PPOConfig):
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
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    return loss, Metrics(loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction)


@partial(jax.jit, static_argnums=(0, 3))
def update_minibatch(model, train_state, batch, config):
    (_, metrics), gradients = jax.value_and_grad(_loss, argnums=1, has_aux=True)(
        model, train_state.params, batch, config
    )
    return train_state.apply_gradients(grads=gradients), metrics


@partial(jax.jit, static_argnums=(0, 4, 5, 6))
def update_epochs(model, train_state, batch, key, config, num_epochs, minibatch_size):
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
                return update_minibatch(model, inner_state, minibatch, config)

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


__all__ = ["PPOBatch", "PPOConfig", "compute_gae", "terminal_win_rewards"]
