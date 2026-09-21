"""Actorを凍結して非対称criticのみを回帰する更新。"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from kaggriculture.policy.common.config import CRITIC_PARAMETER_MODULES
from kaggriculture.policy.jax import policy as P


def create_train_state(
    model, variables, *, learning_rate: float | optax.Schedule, max_grad_norm: float
):
    """Actor leafを厳密に固定したcritic専用optimizerを作る。"""
    params = variables["params"]
    labels = jax.tree_util.tree_map_with_path(
        lambda path, _: "critic" if path[0].key in CRITIC_PARAMETER_MODULES else "actor", params
    )
    optimizer = optax.multi_transform(
        {
            "critic": optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate)
            ),
            "actor": optax.set_to_zero(),
        },
        labels,
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


def update(model, state, states, targets, turns_per_day: int):
    """両席の割引済み終端勝敗へcriticを回帰する。"""

    def loss_fn(params):
        prediction = P.state_values(model, {"params": params}, states, turns_per_day=turns_per_day)
        return jnp.mean(jnp.square(prediction - targets))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def evaluate(model, params, states, targets, turns_per_day: int):
    """価値教師に対する平均二乗誤差を返す。"""
    prediction = P.state_values(model, {"params": params}, states, turns_per_day=turns_per_day)
    return jnp.mean(jnp.square(prediction - targets))
