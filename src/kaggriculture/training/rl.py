"""Shared reinforcement-learning returns and reward utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp


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
