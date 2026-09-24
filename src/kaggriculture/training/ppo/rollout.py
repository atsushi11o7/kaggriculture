"""V3方策・Executor・シミュレータを結ぶGPU完結rollout。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture.policy.jax import history as H
from kaggriculture.policy.jax import policy as P
from kaggriculture.policy.jax import tokenize as T
from kaggriculture.policy.jax.types import Intent
from kaggriculture.simulator.action import Action
from kaggriculture.simulator.reset import reset
from kaggriculture.simulator.state import State
from kaggriculture.simulator.step import step_batch_lockstep
from kaggriculture.training.ppo.core import (
    CriticBatch,
    PPOBatch,
    compute_gae,
    terminal_win_rewards,
)
from kaggriculture.training.rl import DailyRewardConfig, daily_asset_rewards, estimated_assets


@dataclass(frozen=True)
class RolloutConfig:
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
    temperature: float = 0.8
    daily_reward_coefficient: float = 0.05
    daily_reward_scale: float = 10000.0
    daily_reward_maximum: float = 0.02


class Rollout(NamedTuple):
    states: State
    intent: Intent
    slot_mask: jnp.ndarray
    slot_log_prob: jnp.ndarray
    value: jnp.ndarray
    rewards: jnp.ndarray
    daily_rewards: jnp.ndarray
    dones: jnp.ndarray
    counters: T.EpisodeCounters
    executor_invalid: jnp.ndarray
    executor_clamped: jnp.ndarray
    executor_invalid_unit: jnp.ndarray
    executor_clamped_unit: jnp.ndarray
    final_state: State
    final_counters: T.EpisodeCounters
    final_margin: jnp.ndarray
    bootstrap_value: jnp.ndarray


def _step(state, action, config):
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


@partial(jax.jit, static_argnums=(0, 2))
def collect_rollout(
    model, variables, config, initial_state, initial_counters, key, initial_margin=None
):
    """1回の方策forwardを含む環境stepを時間軸scanする。"""
    batch_size = initial_state.step.shape[0]
    if initial_margin is None:
        assets = estimated_assets(initial_state)
        initial_margin = assets[:, 0] - assets[:, 1]
    daily_config = DailyRewardConfig(
        config.daily_reward_coefficient, config.daily_reward_scale, config.daily_reward_maximum
    )

    def scan_step(carry, _):
        state, counters, margin, rng = carry
        rng, action_key, reset_key = jax.random.split(rng, 3)
        output = P.sample_self_play_actions(
            model,
            variables,
            state,
            action_key,
            counters=counters,
            temperature=config.temperature,
            turns_per_day=config.turns_per_day,
            shed_capacity=config.shed_capacity,
            hire_mult=config.hire_mult,
        )
        stepped, cash, done = _step(state, output.action, config)
        daily_rewards, next_margin = daily_asset_rewards(
            stepped, margin, done, config.turns_per_day, daily_config
        )
        rewards = terminal_win_rewards(cash, done) + daily_rewards
        updated_counters = H.update_counters(
            state,
            output.action,
            counters,
            turns_per_day=config.turns_per_day,
            shed_capacity=config.shed_capacity,
            hire_mult=config.hire_mult,
        )
        fresh = reset(
            reset_key,
            batch_size,
            board_size=config.board_size,
            starting_money=config.starting_money,
        )
        next_state = jax.lax.cond(done[0], lambda: fresh, lambda: stepped)
        next_counters = jax.lax.cond(done[0], lambda: H.zeros(batch_size), lambda: updated_counters)
        next_margin = jnp.where(done, 0.0, next_margin)
        transition = (
            state,
            output.intent,
            output.slot_mask,
            output.slot_log_prob,
            output.value,
            rewards,
            daily_rewards,
            done,
            counters,
            output.stats.invalid_market,
            output.stats.clamped_market_quantity,
            output.stats.invalid_unit,
            output.stats.clamped_unit_quantity,
        )
        return (next_state, next_counters, next_margin, rng), transition

    (final_state, final_counters, final_margin, _), values = jax.lax.scan(
        scan_step,
        (initial_state, initial_counters, initial_margin, key),
        xs=None,
        length=config.horizon,
    )
    bootstrap = P.state_values(
        model,
        variables,
        final_state,
        final_counters,
        config.turns_per_day,
    )
    return Rollout(*values, final_state, final_counters, final_margin, bootstrap)


def _stack_seats(seat0, seat1):
    return jax.tree.map(lambda a, b: jnp.stack([a, b], axis=1), seat0, seat1)


@partial(jax.jit, static_argnums=(0, 3, 4))
def collect_rollout_vs_opponent(
    model,
    learner_variables,
    opponent_variables,
    learner_seat: int,
    config,
    initial_state,
    initial_counters,
    key,
    initial_margin=None,
):
    """learner_seat側だけlearner_variablesで行動し、もう片方は固定の
    opponent_variablesで行動する。opponentは学習対象ではないため、
    opponent側のslot_maskは常にFalseで書き出し、PPOの損失に一切寄与
    させない(固定方策の行動をlearnerの現在paramsで評価するのは不整合
    なため)。Rollout/PPOBatch/GAEの形状・計算はcollect_rolloutと共通の
    まま流用できる(マスクされた行は損失への寄与が0になるだけ)。
    """
    batch_size = initial_state.step.shape[0]
    opponent_seat = 1 - learner_seat
    if initial_margin is None:
        assets = estimated_assets(initial_state)
        initial_margin = assets[:, 0] - assets[:, 1]
    daily_config = DailyRewardConfig(
        config.daily_reward_coefficient, config.daily_reward_scale, config.daily_reward_maximum
    )

    def scan_step(carry, _):
        state, counters, margin, rng = carry
        rng, learner_key, opponent_key, reset_key = jax.random.split(rng, 4)
        learner_players = jnp.full((batch_size,), learner_seat, jnp.int32)
        opponent_players = jnp.full((batch_size,), opponent_seat, jnp.int32)
        learner_counters = jax.tree.map(lambda value: value[:, learner_seat], counters)
        opponent_counters = jax.tree.map(lambda value: value[:, opponent_seat], counters)
        learner_out = P.sample_actions(
            model,
            learner_variables,
            state,
            learner_players,
            learner_key,
            learner_counters,
            temperature=config.temperature,
            turns_per_day=config.turns_per_day,
            shed_capacity=config.shed_capacity,
            hire_mult=config.hire_mult,
        )
        opponent_out = P.sample_actions(
            model,
            opponent_variables,
            state,
            opponent_players,
            opponent_key,
            opponent_counters,
            temperature=config.temperature,
            turns_per_day=config.turns_per_day,
            shed_capacity=config.shed_capacity,
            hire_mult=config.hire_mult,
        )
        outputs = [None, None]
        outputs[learner_seat] = learner_out
        outputs[opponent_seat] = opponent_out
        action = Action(
            *(
                jnp.stack([f0, f1], axis=1)
                for f0, f1 in zip(outputs[0].action, outputs[1].action, strict=True)
            )
        )
        stepped, cash, done = _step(state, action, config)
        daily_rewards, next_margin = daily_asset_rewards(
            stepped, margin, done, config.turns_per_day, daily_config
        )
        rewards = terminal_win_rewards(cash, done) + daily_rewards
        updated_counters = H.update_counters(
            state,
            action,
            counters,
            turns_per_day=config.turns_per_day,
            shed_capacity=config.shed_capacity,
            hire_mult=config.hire_mult,
        )
        fresh = reset(
            reset_key,
            batch_size,
            board_size=config.board_size,
            starting_money=config.starting_money,
        )
        next_state = jax.lax.cond(done[0], lambda: fresh, lambda: stepped)
        next_counters = jax.lax.cond(done[0], lambda: H.zeros(batch_size), lambda: updated_counters)
        next_margin = jnp.where(done, 0.0, next_margin)
        intent = _stack_seats(outputs[0].intent, outputs[1].intent)
        slot_masks = [outputs[0].slot_mask, outputs[1].slot_mask]
        slot_masks[opponent_seat] = jnp.zeros_like(slot_masks[opponent_seat])
        slot_mask = jnp.stack(slot_masks, axis=1)
        slot_log_prob = jnp.stack([outputs[0].slot_log_prob, outputs[1].slot_log_prob], axis=1)
        value = jnp.stack([outputs[0].value, outputs[1].value], axis=1)
        invalid = jnp.stack(
            [outputs[0].stats.invalid_market, outputs[1].stats.invalid_market], axis=1
        )
        clamped = jnp.stack(
            [outputs[0].stats.clamped_market_quantity, outputs[1].stats.clamped_market_quantity],
            axis=1,
        )
        invalid_unit = jnp.stack(
            [outputs[0].stats.invalid_unit, outputs[1].stats.invalid_unit], axis=1
        )
        clamped_unit = jnp.stack(
            [outputs[0].stats.clamped_unit_quantity, outputs[1].stats.clamped_unit_quantity],
            axis=1,
        )
        transition = (
            state,
            intent,
            slot_mask,
            slot_log_prob,
            value,
            rewards,
            daily_rewards,
            done,
            counters,
            invalid,
            clamped,
            invalid_unit,
            clamped_unit,
        )
        return (next_state, next_counters, next_margin, rng), transition

    (final_state, final_counters, final_margin, _), values = jax.lax.scan(
        scan_step,
        (initial_state, initial_counters, initial_margin, key),
        xs=None,
        length=config.horizon,
    )
    bootstrap = P.state_values(
        model,
        learner_variables,
        final_state,
        final_counters,
        config.turns_per_day,
    )
    return Rollout(*values, final_state, final_counters, final_margin, bootstrap)


def to_ppo_batch(rollout: Rollout, gamma=0.999, gae_lambda=0.95):
    advantages, returns = compute_gae(
        rollout.rewards, rollout.value, rollout.dones, rollout.bootstrap_value, gamma, gae_lambda
    )
    horizon, batch_size = rollout.dones.shape
    samples = horizon * batch_size * 2
    states = jax.tree.map(
        lambda value: jnp.repeat(value[:, :, None], 2, axis=2).reshape(
            (samples,) + value.shape[2:]
        ),
        rollout.states,
    )
    players = jnp.broadcast_to(jnp.asarray([0, 1]), (horizon, batch_size, 2)).reshape(-1)

    def flatten(value):
        return value.reshape((samples,) + value.shape[3:])

    counters = jax.tree.map(lambda value: value.reshape(samples, -1), rollout.counters)
    return PPOBatch(
        states,
        players,
        jax.tree.map(flatten, rollout.intent),
        flatten(rollout.slot_mask),
        flatten(rollout.slot_log_prob),
        rollout.value.reshape(-1),
        advantages.reshape(-1),
        returns.reshape(-1),
        counters,
    )


def monte_carlo_returns(
    rewards: jnp.ndarray, dones: jnp.ndarray, gamma: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """rollout内で終端結果まで観測できた局面の割引returnを返す。

    rollout末尾の未完了episodeはbootstrapせず、valid=Falseとしてcritic
    warm-upから除外する。終端後にresetされた新episodeの途中も同様に除外する。

    Args:
        rewards: [horizon, batch, 2]の報酬。
        dones: [horizon, batch]の終端フラグ。
        gamma: 割引率。

    Returns:
        (returns, valid)。returnsはrewardsと同shape、validは[horizon, batch]。
    """

    def reverse_step(carry, transition):
        running, has_terminal = carry
        reward, done = transition
        running = reward + gamma * jnp.where(done[:, None], 0.0, running)
        has_terminal = done | (~done & has_terminal)
        return (running, has_terminal), (running, has_terminal)

    initial = (
        jnp.zeros_like(rewards[0]),
        jnp.zeros_like(dones[0], dtype=bool),
    )
    _, (returns, valid) = jax.lax.scan(
        reverse_step,
        initial,
        (rewards, dones),
        reverse=True,
    )
    return returns, valid


def to_critic_batch(rollout: Rollout, gamma: float = 0.999) -> CriticBatch:
    """rolloutを状態単位のcritic warm-up batchへ変換する。"""
    returns, valid = monte_carlo_returns(rollout.rewards, rollout.dones, gamma)
    horizon, batch_size = rollout.dones.shape
    samples = horizon * batch_size
    states = jax.tree.map(
        lambda value: value.reshape((samples,) + value.shape[2:]),
        rollout.states,
    )
    counters = jax.tree.map(
        lambda value: value.reshape((samples,) + value.shape[2:]),
        rollout.counters,
    )
    learner = rollout.slot_mask.any(axis=-1)
    mask = learner & valid[..., None]
    return CriticBatch(
        states,
        returns.reshape(samples, 2),
        mask.reshape(samples, 2),
        counters,
    )
