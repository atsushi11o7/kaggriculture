"""固定slot方策のJAX Behavior Cloning。"""

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
    """BC更新の設定。

    Attributes:
        learning_rate: AdamWのlearning rateまたはschedule。
        weight_decay: AdamWのweight decay係数。
        max_grad_norm: gradient clippingのglobal norm上限。
        turns_per_day: 観測token化に使う1日あたりのturn数。
        shed_capacity: 合法手判定に使うshed容量。
        value_loss_coefficient: policy lossへ加えるvalue lossの係数。
    """

    learning_rate: float | optax.Schedule = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    turns_per_day: int = 24
    shed_capacity: int = 100
    value_loss_coefficient: float = 0.0


class BCMetrics(NamedTuple):
    """1 batchのBC評価値。

    Attributes:
        loss: optimizerへ渡す合計loss。
        policy_loss: 有効slotで平均した負のlog probability。
        value_loss: value予測と教師returnの平均二乗誤差。
        count: policy lossへ使用したslot数。
        excluded: 再評価時に無効と判定され、除外したslot数。
    """

    loss: jnp.ndarray
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    count: jnp.ndarray
    excluded: jnp.ndarray


def create_train_state(model, variables, config: BCConfig):
    """BC用optimizerを持つTrainStateを作る。

    Args:
        model: 学習するFlax方策モデル。
        variables: 初期化済みのモデルvariables。
        config: optimizerを含むBC設定。

    Returns:
        gradient clipping付きAdamWを持つTrainState。
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
    )
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optimizer)


def loss(model, params, batch, config: BCConfig):
    """固定slotのpolicy lossと任意のvalue lossを計算する。

    Args:
        model: 評価するFlax方策モデル。
        params: 評価対象のモデルparameter。
        batch: 状態、教師intent、slot mask、value教師を含むBCBatch。
        config: 環境定数とvalue loss係数を含むBC設定。

    Returns:
        合計lossと内訳を持つBCMetrics。
    """
    evaluation = P.evaluate_intent(
        model,
        {"params": params},
        batch.states,
        batch.players,
        batch.intent,
        counters=batch.counters,
        turns_per_day=config.turns_per_day,
        shed_capacity=config.shed_capacity,
    )
    mask = batch.slot_mask.astype(jnp.float32)
    excluded_mask = batch.slot_mask & ~evaluation.slot_valid
    excluded = jnp.sum(excluded_mask.astype(jnp.float32))
    mask = mask * evaluation.slot_valid.astype(jnp.float32)
    count = jnp.maximum(mask.sum(), 1)
    policy_loss = -jnp.sum(jnp.where(mask > 0, evaluation.slot_log_prob, 0.0)) / count

    # 同じ1回の順伝播で得たcritic出力を、同じ(試合・step・プレイヤー)の教師と比べる。
    value_loss = jnp.asarray(0.0)
    if config.value_loss_coefficient > 0:
        value_loss = jnp.mean(jnp.square(evaluation.value - batch.value_target))

    total = policy_loss + config.value_loss_coefficient * value_loss
    return BCMetrics(total, policy_loss, value_loss, count, excluded)


@partial(jax.jit, static_argnums=(0, 3))
def update_minibatch(model, train_state, batch, config):
    """1 minibatchの勾配を計算してBC parameterを更新する。

    Args:
        model: 学習するFlax方策モデル。
        train_state: 更新前のTrainState。
        batch: 1 minibatch分のBCBatch。
        config: BC設定。

    Returns:
        更新後のTrainStateとBCMetrics。
    """

    def objective(params):
        metrics = loss(model, params, batch, config)
        return metrics.loss, metrics

    (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(train_state.params)
    return train_state.apply_gradients(grads=gradients), metrics


@partial(jax.jit, static_argnums=(0, 3))
def evaluate_minibatch(model, params, batch, config):
    """parameterを更新せず、1 minibatchのBC指標を返す。

    Args:
        model: 評価するFlax方策モデル。
        params: 評価対象のモデルparameter。
        batch: 1 minibatch分のBCBatch。
        config: BC設定。

    Returns:
        batch全体のBCMetrics。
    """
    return loss(model, params, batch, config)
