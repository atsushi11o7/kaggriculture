"""JAX PPOのGAE・目的関数・1 minibatch更新。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from kaggriculture.policy.jax import distribution as D
from kaggriculture.policy.jax import model as M
from kaggriculture.policy.jax import tokenize as T
from kaggriculture.simulator.state import State


@dataclass(frozen=True)
class PPOConfig:
    """PPO更新のハイパーパラメータ。"""

    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    temperature: float = 1.0
    ratio_mode: str = "token"
    turns_per_day: int = 24
    shed_capacity: int = 100
    hire_mult: float = 1.0


class PPOBatch(NamedTuple):
    """環境・時刻・player軸を平坦化したPPO minibatch。"""

    states: State
    players: jnp.ndarray
    choices: jnp.ndarray
    decision_mask: jnp.ndarray
    old_token_log_prob: jnp.ndarray
    old_log_prob: jnp.ndarray
    old_value: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    counters: T.EpisodeCounters | None = None


class PPOMetrics(NamedTuple):
    loss: jnp.ndarray
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    entropy: jnp.ndarray
    approx_kl: jnp.ndarray
    clip_fraction: jnp.ndarray


def terminal_win_rewards(cash: jnp.ndarray, done: jnp.ndarray) -> jnp.ndarray:
    """終端所持金をゼロ和の勝敗報酬(+1/-1、同点0)へ変換する。"""
    margin = cash[..., 0] - cash[..., 1]
    result = jnp.sign(margin)
    rewards = jnp.stack([result, -result], axis=-1)
    return jnp.where(done[..., None], rewards, 0.0)


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """時間先頭軸のrolloutからGAE advantageとreturnを計算する。"""
    if dones.ndim == rewards.ndim - 1:
        dones = dones[..., None]
    next_values = jnp.concatenate([values[1:], bootstrap_value[None]], axis=0)

    def backward(advantage, inputs):
        reward, value, next_value, done = inputs
        discount = gamma * (1.0 - done.astype(jnp.float32))
        delta = reward + discount * next_value - value
        advantage = delta + discount * gae_lambda * advantage
        return advantage, advantage

    _, reversed_advantages = jax.lax.scan(
        backward,
        jnp.zeros_like(bootstrap_value),
        (rewards[::-1], values[::-1], next_values[::-1], dones[::-1]),
    )
    advantages = reversed_advantages[::-1]
    return advantages, advantages + values


def create_train_state(model: M.PolicyValueNet, variables: dict, config: PPOConfig) -> TrainState:
    """gradient clipping付きAdamWの学習状態を作る。"""
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
    )


def _loss(
    model: M.PolicyValueNet,
    params: dict,
    batch: PPOBatch,
    config: PPOConfig,
) -> tuple[jnp.ndarray, PPOMetrics]:
    evaluation = D.evaluate_choices(
        model,
        {"params": params},
        batch.states,
        batch.players,
        batch.choices,
        counters=batch.counters,
        temperature=config.temperature,
        turns_per_day=config.turns_per_day,
        shed_capacity=config.shed_capacity,
        hire_mult=config.hire_mult,
    )
    advantages = (batch.advantages - jnp.mean(batch.advantages)) / (
        jnp.std(batch.advantages) + 1e-8
    )
    if config.ratio_mode == "token":
        log_ratio = evaluation.token_log_prob - batch.old_token_log_prob
        ratio = jnp.exp(jnp.clip(log_ratio, -20.0, 20.0))
        token_advantages = advantages[:, None]
        unclipped = ratio * token_advantages
        clipped = (
            jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * token_advantages
        )
        mask = batch.decision_mask.astype(jnp.float32)
        decisions_per_sample = jnp.maximum(jnp.sum(mask, axis=-1), 1.0)
        policy_per_sample = (
            jnp.sum(jnp.minimum(unclipped, clipped) * mask, axis=-1) / decisions_per_sample
        )
        policy_loss = -jnp.mean(policy_per_sample)
        approx_kl = jnp.mean(
            jnp.sum(((ratio - 1.0) - log_ratio) * mask, axis=-1) / decisions_per_sample
        )
        clip_fraction = jnp.mean(
            jnp.sum(
                (jnp.abs(ratio - 1.0) > config.clip_epsilon).astype(jnp.float32) * mask,
                axis=-1,
            )
            / decisions_per_sample
        )
    else:
        log_ratio = evaluation.log_prob - batch.old_log_prob
        ratio = jnp.exp(jnp.clip(log_ratio, -20.0, 20.0))
        unclipped = ratio * advantages
        clipped = jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advantages
        policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
        approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
        clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > config.clip_epsilon).astype(jnp.float32))

    value_delta = evaluation.value - batch.old_value
    clipped_value = batch.old_value + jnp.clip(
        value_delta, -config.value_clip_epsilon, config.value_clip_epsilon
    )
    value_error = jnp.square(evaluation.value - batch.returns)
    clipped_value_error = jnp.square(clipped_value - batch.returns)
    value_loss = 0.5 * jnp.mean(jnp.maximum(value_error, clipped_value_error))

    entropy = jnp.mean(evaluation.entropy / jnp.maximum(evaluation.num_decisions, 1))
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    metrics = PPOMetrics(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
    )
    return loss, metrics


@partial(jax.jit, static_argnums=(0, 3))
def update_minibatch(
    model: M.PolicyValueNet,
    train_state: TrainState,
    batch: PPOBatch,
    config: PPOConfig,
) -> tuple[TrainState, PPOMetrics]:
    """1 minibatchのPPO勾配更新を行う。"""
    (_, metrics), gradients = jax.value_and_grad(_loss, argnums=1, has_aux=True)(
        model, train_state.params, batch, config
    )
    return train_state.apply_gradients(grads=gradients), metrics


@partial(jax.jit, static_argnums=(0, 4, 5, 6))
def update_epochs(
    model: M.PolicyValueNet,
    train_state: TrainState,
    batch: PPOBatch,
    key: jax.Array,
    config: PPOConfig,
    num_epochs: int,
    minibatch_size: int,
) -> tuple[TrainState, PPOMetrics]:
    """rolloutをepochごとにshuffleし、全minibatchをGPU上で更新する。"""
    sample_size = batch.players.shape[0]
    if sample_size % minibatch_size:
        raise ValueError("sample count must be divisible by minibatch_size")
    num_minibatches = sample_size // minibatch_size

    def epoch_step(carry, _):
        state, rng = carry
        rng, permutation_key = jax.random.split(rng)
        indices = jax.random.permutation(permutation_key, sample_size).reshape(
            num_minibatches, minibatch_size
        )

        def minibatch_step(current_state, sample_indices):
            minibatch = jax.tree.map(
                lambda value: value[sample_indices] if value is not None else None,
                batch,
                is_leaf=lambda value: value is None,
            )
            return update_minibatch(model, current_state, minibatch, config)

        state, metrics = jax.lax.scan(minibatch_step, state, indices)
        return (state, rng), jax.tree.map(jnp.mean, metrics)

    (train_state, _), metrics = jax.lax.scan(
        epoch_step, (train_state, key), xs=None, length=num_epochs
    )
    return train_state, jax.tree.map(jnp.mean, metrics)
