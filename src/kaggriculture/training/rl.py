"""Shared reinforcement-learning returns and reward utilities."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from kaggriculture.simulator.assets import estimated_assets
from kaggriculture.simulator.state import State


@dataclass(frozen=True)
class DailyRewardConfig:
    """日末の資産差改善に与える補助報酬の設定。"""

    coefficient: float = 0.05
    scale: float = 10000.0
    maximum: float = 0.02

    def __post_init__(self) -> None:
        if self.coefficient < 0 or self.scale <= 0 or self.maximum <= 0:
            raise ValueError("daily reward requires nonnegative coefficient and positive scale/max")


def daily_asset_rewards(
    state: State,
    previous_margin: jnp.ndarray,
    done: jnp.ndarray,
    turns_per_day: int,
    config: DailyRewardConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """日末または終端で、前回評価時からの資産差改善をゼロ和報酬にする。

    Args:
        state: 当該ターンを実行した後の状態。
        previous_margin: 前回評価時の自分対相手の概算資産差。
        done: 試合終端フラグ。
        turns_per_day: 1日当たりのターン数。
        config: 補助報酬の係数と上限。

    Returns:
        (両プレイヤーの補助報酬、次回評価用の資産差)。
    """
    boundary = ((state.step % turns_per_day) == 0) | done
    if config.coefficient == 0:
        return jnp.zeros((state.step.shape[0], 2), dtype=jnp.float32), previous_margin

    def evaluate():
        assets = estimated_assets(state)
        margin = assets[:, 0] - assets[:, 1]
        change = margin - previous_margin
        reward = jnp.clip(
            config.coefficient * change / config.scale,
            -config.maximum,
            config.maximum,
        )
        reward = jnp.where(boundary, reward, 0.0)
        next_margin = jnp.where(boundary, margin, previous_margin)
        return jnp.stack([reward, -reward], axis=-1), next_margin

    return jax.lax.cond(
        jnp.any(boundary),
        evaluate,
        lambda: (jnp.zeros((state.step.shape[0], 2), dtype=jnp.float32), previous_margin),
    )


def terminal_win_rewards(cash: jnp.ndarray, done: jnp.ndarray) -> jnp.ndarray:
    """Convert terminal cash margins to zero-sum win rewards."""
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
    """Compute GAE advantages and returns along the leading time axis."""
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
