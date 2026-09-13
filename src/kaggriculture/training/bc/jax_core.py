"""固定shape replay batchを使うJAX Behavior Cloning更新。"""

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
from kaggriculture.training.bc.jax_cache import BCBatch


@dataclass(frozen=True)
class BCConfig:
    """JAX BC更新の設定。"""

    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    turns_per_day: int
    shed_capacity: int
    hire_mult: float


class BCMetrics(NamedTuple):
    """1更新のtoken平均指標。"""

    loss: jnp.ndarray
    entropy: jnp.ndarray
    num_tokens: jnp.ndarray


def create_train_state(model: M.PolicyValueNet, variables: dict, config: BCConfig) -> TrainState:
    """gradient clipping付きAdamWを作る。"""
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
    )
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optimizer)


def _loss(
    model: M.PolicyValueNet,
    params: dict,
    batch: BCBatch,
    config: BCConfig,
) -> tuple[jnp.ndarray, BCMetrics]:
    evaluation = D.evaluate_choices(
        model,
        {"params": params},
        batch.states,
        batch.players,
        batch.choices,
        turns_per_day=config.turns_per_day,
        shed_capacity=config.shed_capacity,
        hire_mult=config.hire_mult,
    )
    mask = batch.decision_mask.astype(jnp.float32)
    num_tokens = jnp.sum(mask)
    loss = -jnp.sum(evaluation.token_log_prob * mask) / jnp.maximum(num_tokens, 1.0)
    entropy = jnp.sum(evaluation.token_entropy * mask) / jnp.maximum(num_tokens, 1.0)
    return loss, BCMetrics(loss, entropy, num_tokens)


@partial(jax.jit, static_argnums=(0, 3))
def update_minibatch(
    model: M.PolicyValueNet,
    train_state: TrainState,
    batch: BCBatch,
    config: BCConfig,
) -> tuple[TrainState, BCMetrics]:
    """1 minibatchをGPU上で更新する。"""
    (_, metrics), gradients = jax.value_and_grad(_loss, argnums=1, has_aux=True)(
        model, train_state.params, batch, config
    )
    return train_state.apply_gradients(grads=gradients), metrics


@partial(jax.jit, static_argnums=(0, 3))
def evaluate_minibatch(
    model: M.PolicyValueNet,
    params: dict,
    batch: BCBatch,
    config: BCConfig,
) -> BCMetrics:
    """1 validation minibatchを決定的に評価する。"""
    _, metrics = _loss(model, params, batch, config)
    return metrics
