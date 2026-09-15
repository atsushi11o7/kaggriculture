"""非自己回帰固定slot方策のJAX Behavior Cloning。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from kaggriculture.policy.jax import policy as P


@dataclass(frozen=True)
class BCConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    turns_per_day: int = 24
    shed_capacity: int = 100


def create_train_state(model, variables, config: BCConfig):
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
    )
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optimizer)


def loss(model, params, batch, config: BCConfig):
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
    count = jnp.maximum(mask.sum(), 1)
    nll = -jnp.sum(jnp.where(mask > 0, evaluation.slot_log_prob, 0.0)) / count
    return nll, count


@partial(jax.jit, static_argnums=(0, 3))
def update_minibatch(model, train_state, batch, config):
    def objective(params):
        return loss(model, params, batch, config)

    (value, count), gradients = jax.value_and_grad(objective, has_aux=True)(train_state.params)
    return train_state.apply_gradients(grads=gradients), (value, count)


@partial(jax.jit, static_argnums=(0, 3))
def evaluate_minibatch(model, params, batch, config):
    return loss(model, params, batch, config)
