"""JAX方策とシミュレータを接続するGPU完結rollout収集。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture.policy.jax import distribution as D
from kaggriculture.policy.jax import history as H
from kaggriculture.policy.jax import model as M
from kaggriculture.policy.jax import tokenize as T
from kaggriculture.simulator.reset import reset
from kaggriculture.simulator.state import State
from kaggriculture.simulator.step import step_batch_lockstep
from kaggriculture.training.ppo.core import PPOBatch, compute_gae, terminal_win_rewards


@dataclass(frozen=True)
class RolloutConfig:
    """rollout長とシミュレータの静的設定。"""

    horizon: int = 128
    board_size: int = 10
    turns_per_day: int = 24
    shed_capacity: int = 100
    weed_chance: float = 0.005
    shop_unlock_interval: int = 3
    shop_sell_interval: int = 4
    center_sell_interval: int = 24
    hire_mult: int = 1
    max_shop_instances: int = 8
    episode_steps: int = 720
    starting_money: float = 3000.0
    temperature: float = 1.0


class Rollout(NamedTuple):
    """時間軸先頭のPPO rollout。"""

    states: State
    choices: jnp.ndarray
    decision_mask: jnp.ndarray
    token_log_prob: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    num_decisions: jnp.ndarray
    counters: T.EpisodeCounters
    final_state: State
    final_counters: T.EpisodeCounters
    bootstrap_value: jnp.ndarray


def _step_simulator(state: State, action, config: RolloutConfig):
    return step_batch_lockstep(
        state,
        action,
        board_size=config.board_size,
        turns_per_day=config.turns_per_day,
        shed_capacity=config.shed_capacity,
        weed_chance=config.weed_chance,
        shop_unlock_interval=config.shop_unlock_interval,
        shop_sell_interval=config.shop_sell_interval,
        center_sell_interval=config.center_sell_interval,
        hire_mult=config.hire_mult,
        max_shop_instances=config.max_shop_instances,
        episode_steps=config.episode_steps,
    )


@partial(jax.jit, static_argnums=(0, 3))
def collect_rollout(
    model: M.PolicyValueNet,
    variables: dict,
    cache_template: dict,
    config: RolloutConfig,
    initial_state: State,
    initial_counters: T.EpisodeCounters,
    key: jax.Array,
) -> Rollout:
    """方策生成・環境遷移・終端resetを1つの`lax.scan`で実行する。"""
    batch_size = initial_state.step.shape[0]

    def scan_step(carry, _):
        state, counters, rng = carry
        rng, action_key, reset_key = jax.random.split(rng, 3)
        output = D.sample_self_play_actions(
            model,
            variables,
            cache_template,
            state,
            action_key,
            counters=counters if model.config.use_episode_history else None,
            temperature=config.temperature,
            turns_per_day=config.turns_per_day,
            shed_capacity=config.shed_capacity,
            hire_mult=config.hire_mult,
        )
        stepped_state, terminal_cash, done = _step_simulator(state, output.action, config)
        reward = terminal_win_rewards(terminal_cash, done)
        if model.config.use_episode_history:
            updated_counters = H.update_counters(
                state,
                output.action,
                counters,
                turns_per_day=config.turns_per_day,
                shed_capacity=config.shed_capacity,
                hire_mult=config.hire_mult,
            )
        else:
            updated_counters = counters
        fresh_state = reset(
            reset_key,
            batch_size,
            board_size=config.board_size,
            starting_money=config.starting_money,
        )
        # lockstep batchなので全局が同じターンで同時に終端へ到達する。
        next_state = jax.lax.cond(done[0], lambda: fresh_state, lambda: stepped_state)
        next_counters = jax.lax.cond(done[0], lambda: H.zeros(batch_size), lambda: updated_counters)
        transition = (
            state,
            output.choices,
            output.decision_mask,
            output.token_log_prob,
            output.log_prob,
            output.value,
            reward,
            done,
            output.num_decisions,
            counters,
        )
        return (next_state, next_counters, rng), transition

    (final_state, final_counters, _), transitions = jax.lax.scan(
        scan_step, (initial_state, initial_counters, key), xs=None, length=config.horizon
    )
    bootstrap_value = D.state_values(
        model,
        variables,
        final_state,
        final_counters if model.config.use_episode_history else None,
        turns_per_day=config.turns_per_day,
    )
    return Rollout(*transitions, final_state, final_counters, bootstrap_value)


def to_ppo_batch(
    rollout: Rollout,
    gamma: float = 0.999,
    gae_lambda: float = 0.95,
) -> PPOBatch:
    """`[time, env, player]` rolloutをPPO用の1次元sample軸へ変換する。"""
    advantages, returns = compute_gae(
        rollout.rewards,
        rollout.value,
        rollout.dones,
        rollout.bootstrap_value,
        gamma,
        gae_lambda,
    )
    horizon, batch_size = rollout.dones.shape
    sample_size = horizon * batch_size * 2

    # 各Stateをplayer 0, 1の順に複製し、行動側の[B, 2]順と揃える。
    states = jax.tree.map(
        lambda value: jnp.repeat(value[:, :, None], 2, axis=2).reshape(
            (sample_size,) + value.shape[2:]
        ),
        rollout.states,
    )
    players = jnp.broadcast_to(jnp.asarray([0, 1]), (horizon, batch_size, 2)).reshape(-1)
    counters = jax.tree.map(lambda value: value.reshape(sample_size, -1), rollout.counters)
    return PPOBatch(
        states=states,
        players=players,
        choices=rollout.choices.reshape(sample_size, -1),
        decision_mask=rollout.decision_mask.reshape(sample_size, -1),
        old_token_log_prob=rollout.token_log_prob.reshape(sample_size, -1),
        old_log_prob=rollout.log_prob.reshape(-1),
        old_value=rollout.value.reshape(-1),
        advantages=advantages.reshape(-1),
        returns=returns.reshape(-1),
        counters=counters,
    )
