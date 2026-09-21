"""非自己回帰固定slot方策のJAX Behavior Cloning。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from kaggriculture.policy.jax import policy as P


@dataclass(frozen=True)
class BCConfig:
    learning_rate: float | optax.Schedule = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    turns_per_day: int = 24
    shed_capacity: int = 100
    value_loss_coefficient: float = 0.0


class BCMetrics(NamedTuple):
    loss: jnp.ndarray
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    count: jnp.ndarray
    excluded: jnp.ndarray


def create_train_state(model, variables, config: BCConfig):
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
    )
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optimizer)


def loss(model, params, batch, config: BCConfig, value_batch=None):
    evaluation = P.evaluate_intent(
        model,
        {"params": params},
        batch.states,
        batch.players,
        batch.intent,
        turns_per_day=config.turns_per_day,
        shed_capacity=config.shed_capacity,
    )
    mask = batch.slot_mask.astype(jnp.float32)
    excluded_mask = batch.slot_mask & ~evaluation.slot_valid
    excluded = jnp.sum(excluded_mask.astype(jnp.float32))
    mask = mask * evaluation.slot_valid.astype(jnp.float32)
    count = jnp.maximum(mask.sum(), 1)
    policy_loss = -jnp.sum(jnp.where(mask > 0, evaluation.slot_log_prob, 0.0)) / count

    value_loss = jnp.asarray(0.0)
    if config.value_loss_coefficient > 0:
        if value_batch is None:
            raise ValueError("value_batch is required when value loss is enabled")
        value_states, value_targets = value_batch
        predictions = P.state_values(
            model,
            {"params": params},
            value_states,
            turns_per_day=config.turns_per_day,
        )
        value_loss = jnp.mean(jnp.square(predictions - value_targets))

    total = policy_loss + config.value_loss_coefficient * value_loss
    return BCMetrics(total, policy_loss, value_loss, count, excluded)


@partial(jax.jit, static_argnums=(0, 3))
def update_minibatch(model, train_state, batch, config, value_batch=None):
    def objective(params):
        metrics = loss(model, params, batch, config, value_batch)
        return metrics.loss, metrics

    (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(train_state.params)
    return train_state.apply_gradients(grads=gradients), metrics


@partial(jax.jit, static_argnums=(0, 3))
def evaluate_minibatch(model, params, batch, config, value_batch=None):
    return loss(model, params, batch, config, value_batch)
