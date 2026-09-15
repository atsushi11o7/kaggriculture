"""異なるvariables同士を対戦させ、座席バイアスを除いた勝率を求める。"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture.policy.jax import history as H
from kaggriculture.policy.jax import policy as P
from kaggriculture.simulator.action import Action
from kaggriculture.simulator.reset import reset
from kaggriculture.simulator.step import step_batch_lockstep
from kaggriculture.training.ppo.rollout import RolloutConfig


class EvaluationResult(NamedTuple):
    """候補視点(cashの列0が常に評価対象)での対局結果。"""

    cash: jnp.ndarray
    outcome: jnp.ndarray
    win_rate: jnp.ndarray


def _win_rate(outcome: jnp.ndarray) -> jnp.ndarray:
    """勝ち=1点、引き分け=0.5点、負け=0点として平均する(Eloの期待勝率と同じ扱い)。

    greedy(決定論的)同士の対称な対局は完全な引き分け(outcome==0)になりやすく、
    mean(outcome > 0)だけだと引き分けを「負け」として扱ってしまい、
    同程度の実力同士でもwin_rateが不当に低く出る(昇格判定が機能しなくなる)。
    """
    return jnp.mean(jnp.where(outcome > 0, 1.0, jnp.where(outcome < 0, 0.0, 0.5)))


def _combine_seat_actions(action0: Action, action1: Action) -> Action:
    """座席毎に別variablesで生成した単一Actionを(batch, 2, ...)へまとめる。"""
    return Action(*(jnp.stack([f0, f1], axis=1) for f0, f1 in zip(action0, action1, strict=True)))


@partial(jax.jit, static_argnums=(0, 3, 5))
def evaluate_closed_loop(
    model,
    seat0_variables: dict,
    seat1_variables: dict,
    config: RolloutConfig,
    key: jax.Array,
    batch_size: int,
) -> EvaluationResult:
    """seat0/seat1それぞれ固定のvariablesで終端まで自走させる。"""
    reset_key, run_key = jax.random.split(key)
    initial_state = reset(
        reset_key, batch_size, board_size=config.board_size, starting_money=config.starting_money
    )
    players0 = jnp.zeros((batch_size,), jnp.int32)
    players1 = jnp.ones((batch_size,), jnp.int32)
    use_history = model.config.use_episode_history
    initial_counters = H.zeros(batch_size) if use_history else None

    def scan_step(carry, step_key):
        state, counters = carry
        key0, key1 = jax.random.split(step_key)
        out0 = P.sample_actions(
            model,
            seat0_variables,
            state,
            players0,
            key0,
            counters[:, 0] if use_history else None,
            temperature=config.temperature,
            greedy=True,
            turns_per_day=config.turns_per_day,
            shed_capacity=config.shed_capacity,
            hire_mult=config.hire_mult,
        )
        out1 = P.sample_actions(
            model,
            seat1_variables,
            state,
            players1,
            key1,
            counters[:, 1] if use_history else None,
            temperature=config.temperature,
            greedy=True,
            turns_per_day=config.turns_per_day,
            shed_capacity=config.shed_capacity,
            hire_mult=config.hire_mult,
        )
        action = _combine_seat_actions(out0.action, out1.action)
        next_state, _, _ = step_batch_lockstep(
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
        next_counters = (
            H.update_counters(
                state,
                action,
                counters,
                turns_per_day=config.turns_per_day,
                shed_capacity=config.shed_capacity,
                hire_mult=config.hire_mult,
            )
            if use_history
            else counters
        )
        return (next_state, next_counters), None

    keys = jax.random.split(run_key, config.episode_steps - 1)
    (final_state, _), _ = jax.lax.scan(scan_step, (initial_state, initial_counters), keys)
    cash = final_state.money
    outcome = jnp.sign(cash[:, 0] - cash[:, 1])
    return EvaluationResult(cash, outcome, _win_rate(outcome))


def _combine_seats(seat0: EvaluationResult, seat1: EvaluationResult) -> EvaluationResult:
    """候補がseat0だった回とseat1だった回を、候補視点1本のcash/outcomeへ揃える。"""
    cash = jnp.concatenate([seat0.cash, seat1.cash[:, ::-1]], axis=0)
    outcome = jnp.concatenate([seat0.outcome, -seat1.outcome], axis=0)
    return EvaluationResult(cash, outcome, _win_rate(outcome))


def evaluate_both_seats(
    model,
    actor_variables: dict,
    opponent_variables: dict,
    config: RolloutConfig,
    key: jax.Array,
    games_per_seat: int,
) -> EvaluationResult:
    """候補をseat0・seat1それぞれで対局させ、座席の有利不利を打ち消す。"""
    key0, key1 = jax.random.split(key)
    as_seat0 = evaluate_closed_loop(
        model, actor_variables, opponent_variables, config, key0, games_per_seat
    )
    as_seat1 = evaluate_closed_loop(
        model, opponent_variables, actor_variables, config, key1, games_per_seat
    )
    return _combine_seats(as_seat0, as_seat1)
