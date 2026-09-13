"""JAXシミュレータ上のclosed-loop対戦評価。"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture.policy.jax import distribution as D
from kaggriculture.policy.jax import history as H
from kaggriculture.policy.jax import model as M
from kaggriculture.simulator.reset import reset
from kaggriculture.simulator.step import step_batch_lockstep
from kaggriculture.training.ppo.rollout import RolloutConfig


class EvaluationResult(NamedTuple):
    """全対局の終端所持金とplayer 0視点の勝敗。"""

    cash: jnp.ndarray
    outcome: jnp.ndarray
    win_rate: jnp.ndarray


@partial(jax.jit, static_argnums=(0, 4, 6))
def evaluate_closed_loop(
    model: M.PolicyValueNet,
    actor_variables: dict,
    opponent_variables: dict,
    cache_template: dict,
    config: RolloutConfig,
    key: jax.Array,
    batch_size: int,
) -> EvaluationResult:
    """actorを固定opponentと終端まで自走させる。教師行動は参照しない。"""
    key, reset_key = jax.random.split(key)
    state = reset(
        reset_key,
        batch_size,
        board_size=config.board_size,
        starting_money=config.starting_money,
    )
    player0 = jnp.zeros((batch_size,), dtype=jnp.int32)
    player1 = jnp.ones((batch_size,), dtype=jnp.int32)
    counters = H.zeros(batch_size)

    def step(carry, rng):
        current_state, current_counters = carry
        key0, key1 = jax.random.split(rng)
        output0 = D.sample_actions(
            model,
            actor_variables,
            cache_template,
            current_state,
            player0,
            key0,
            counters=jax.tree.map(lambda value: value[:, 0], current_counters)
            if model.config.use_episode_history
            else None,
            temperature=config.temperature,
            greedy=True,
            turns_per_day=config.turns_per_day,
            shed_capacity=config.shed_capacity,
            hire_mult=config.hire_mult,
        )
        output1 = D.sample_actions(
            model,
            opponent_variables,
            cache_template,
            current_state,
            player1,
            key1,
            counters=jax.tree.map(lambda value: value[:, 1], current_counters)
            if model.config.use_episode_history
            else None,
            temperature=config.temperature,
            greedy=True,
            turns_per_day=config.turns_per_day,
            shed_capacity=config.shed_capacity,
            hire_mult=config.hire_mult,
        )
        action = D.combine_player_actions(output0.action, output1.action)
        next_state, _, _ = step_batch_lockstep(
            current_state,
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
        if model.config.use_episode_history:
            next_counters = H.update_counters(
                current_state,
                action,
                current_counters,
                turns_per_day=config.turns_per_day,
                shed_capacity=config.shed_capacity,
                hire_mult=config.hire_mult,
            )
        else:
            next_counters = current_counters
        return (next_state, next_counters), None

    keys = jax.random.split(key, config.episode_steps - 1)
    (final_state, _), _ = jax.lax.scan(step, (state, counters), keys)
    outcome = jnp.sign(final_state.money[:, 0] - final_state.money[:, 1])
    return EvaluationResult(
        cash=final_state.money,
        outcome=outcome,
        win_rate=jnp.mean((outcome > 0).astype(jnp.float32)),
    )
