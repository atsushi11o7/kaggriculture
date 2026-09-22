"""完了試合を蓄積し、勾配を1回だけ適用するPPO更新。

従来は、128ターンずつ切ったrolloutごとにモデルを更新していた。この方式は、

- 勝敗が入るのは約5.6 updateに1回で、それ以外はcriticの予測の差だけがadvantageになる
- 1 updateの経験が少なく、方策勾配の向きがrollout間でほぼ揃わない(ノイズ支配)

という問題があった。ここでは、全環境を初期状態から720ターン最後まで回して完了試合を集め、
試合全体で最後から後ろへadvantageを計算し、複数試合分の勾配を足してから1回だけ更新する。

状態は全ターン分をメモリに置けないため、次の2段階にする。

1. 方策で試合を回し、行動・報酬・価値などの小さい配列だけを保存する。
2. 保存した行動でシミュレータだけを再度回して状態を再生成し、区間ごとに勾配を蓄積する。
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from kaggriculture.policy.jax import policy as P
from kaggriculture.policy.jax.types import Intent
from kaggriculture.simulator.action import Action
from kaggriculture.simulator.reset import reset
from kaggriculture.training.ppo import core
from kaggriculture.training.ppo.rollout import RolloutConfig, _step
from kaggriculture.training.rl import (
    DailyRewardConfig,
    compute_gae,
    daily_asset_rewards,
    estimated_assets,
    terminal_win_rewards,
)


class CompactGame(NamedTuple):
    """1ラウンド(全環境の完了試合)の、状態を含まない記録。全配列の先頭軸は[ターン, 環境]。"""

    actions: Action
    intent: Intent
    slot_mask: jnp.ndarray
    slot_log_prob: jnp.ndarray
    value: jnp.ndarray
    rewards: jnp.ndarray
    daily_rewards: jnp.ndarray
    dones: jnp.ndarray
    invalid_market: jnp.ndarray
    clamped_market: jnp.ndarray
    invalid_unit: jnp.ndarray
    clamped_unit: jnp.ndarray
    final_cash: jnp.ndarray


def initial_state(reset_key, batch_size: int, config: RolloutConfig):
    """回収と再生成で同じ初期状態を作る。"""
    return reset(
        reset_key,
        batch_size,
        board_size=config.board_size,
        starting_money=config.starting_money,
    )


@partial(jax.jit, static_argnums=(0, 3, 4, 5))
def collect_compact_game(
    model,
    learner_variables,
    opponent_variables,
    learner_seat: int,
    config: RolloutConfig,
    batch_size: int,
    reset_key,
    run_key,
) -> CompactGame:
    """全環境を初期状態から最後まで回し、状態を保存せずに行動と統計だけを返す。"""
    if model.config.use_episode_history:
        raise NotImplementedError("full-game PPO does not support episode history")
    opponent_seat = 1 - learner_seat
    state = initial_state(reset_key, batch_size, config)
    assets = estimated_assets(state)
    margin = assets[:, 0] - assets[:, 1]
    daily_config = DailyRewardConfig(
        config.daily_reward_coefficient, config.daily_reward_scale, config.daily_reward_maximum
    )
    learner_players = jnp.full((batch_size,), learner_seat, jnp.int32)
    opponent_players = jnp.full((batch_size,), opponent_seat, jnp.int32)

    def scan_step(carry, step_key):
        state, margin = carry
        learner_key, opponent_key = jax.random.split(step_key)
        learner_out = P.sample_actions(
            model,
            learner_variables,
            state,
            learner_players,
            learner_key,
            None,
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
            None,
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
        daily, next_margin = daily_asset_rewards(
            stepped, margin, done, config.turns_per_day, daily_config
        )
        rewards = terminal_win_rewards(cash, done) + daily
        record = (
            action,
            learner_out.intent,
            learner_out.slot_mask,
            learner_out.slot_log_prob,
            learner_out.value,
            rewards[:, learner_seat],
            daily[:, learner_seat],
            done,
            learner_out.stats.invalid_market,
            learner_out.stats.clamped_market_quantity,
            learner_out.stats.invalid_unit,
            learner_out.stats.clamped_unit_quantity,
            cash,
        )
        return (stepped, next_margin), record

    keys = jax.random.split(run_key, config.episode_steps - 1)
    _, records = jax.lax.scan(scan_step, (state, margin), keys)
    *fields, cash = records
    return CompactGame(*fields, final_cash=cash[-1])


def game_targets(
    game: CompactGame, gamma: float, gae_lambda: float, value_lambda: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """試合全体で、最後のターンから後ろへadvantageと価値の教師を計算する。

    完了した試合なので、終端の次の状態の価値は0で、価値の教師(value_lambda=1)は、criticの予測を
    含まない実際のreturnになる。方策のadvantageには、別のgae_lambdaを使える。
    """
    rewards = game.rewards[..., None]
    values = game.value[..., None]
    bootstrap = jnp.zeros((rewards.shape[1], 1), rewards.dtype)
    advantages, _ = compute_gae(rewards, values, game.dones, bootstrap, gamma, gae_lambda)
    _, returns = compute_gae(rewards, values, game.dones, bootstrap, gamma, value_lambda)
    return advantages[..., 0], returns[..., 0]


def segment_bounds(steps: int, length: int) -> list[tuple[int, int]]:
    """全ターンを、length以下の連続区間に分ける。"""
    return [(start, min(start + length, steps)) for start in range(0, steps, length)]


@partial(jax.jit, static_argnums=(0,))
def replay_segment(config: RolloutConfig, state, actions: Action):
    """保存した行動でシミュレータだけを進め、各ターンの行動前の状態を再生成する。"""

    def step(state, action):
        stepped, _, _ = _step(state, action, config)
        return stepped, state

    return jax.lax.scan(step, state, actions)


@partial(jax.jit, static_argnums=(0, 5, 6, 7))
def segment_gradient(
    model,
    params,
    reference_params,
    states,
    rows: dict,
    ppo_config: core.PPOConfig,
    minibatch: int,
    learner_seat: int,
    permutation_key,
):
    """1区間の全minibatchの勾配とmetricを足し合わせる。

    区間内の行をシャッフルし、minibatchに満たない端数は捨てる。
    """
    steps, batch = rows["slot_mask"].shape[:2]
    total = steps * batch

    def flatten(value):
        return value.reshape((total,) + value.shape[2:])

    flat_states = jax.tree.map(flatten, states)
    flat_rows = jax.tree.map(flatten, rows)
    usable = (total // minibatch) * minibatch
    order = jax.random.permutation(permutation_key, total)[:usable].reshape(-1, minibatch)

    def minibatch_step(carry, index):
        gradient_sum, metric_sum = carry
        take = lambda value: value[index]  # noqa: E731
        selected = jax.tree.map(take, flat_rows)
        batch_ = core.PPOBatch(
            jax.tree.map(take, flat_states),
            jnp.full((minibatch,), learner_seat, jnp.int32),
            selected["intent"],
            selected["slot_mask"],
            selected["old_slot_log_prob"],
            selected["old_value"],
            selected["advantages"],
            selected["returns"],
            None,
        )
        (_, metrics), gradients = jax.value_and_grad(core._loss, argnums=1, has_aux=True)(
            model, params, batch_, ppo_config, reference_params
        )
        gradient_sum = jax.tree.map(jnp.add, gradient_sum, gradients)
        metric_sum = jax.tree.map(jnp.add, metric_sum, metrics)
        return (gradient_sum, metric_sum), None

    zero_metrics = core.Metrics(*(jnp.zeros(()) for _ in core.Metrics._fields))
    init = (jax.tree.map(jnp.zeros_like, params), zero_metrics)
    (gradient_sum, metric_sum), _ = jax.lax.scan(minibatch_step, init, order)
    return gradient_sum, metric_sum, order.shape[0]


def _window(start: int, end: int):
    """先頭軸(ターン)の[start, end)を、pytree全体から切り出す関数を返す。"""
    return lambda tree: jax.tree.map(lambda value: value[start:end], tree)


def _cosine(a, b) -> float:
    leaves_a, leaves_b = jax.tree.leaves(a), jax.tree.leaves(b)
    dot = sum(jnp.sum(x * y) for x, y in zip(leaves_a, leaves_b, strict=True))
    norm_a = jnp.sqrt(sum(jnp.sum(x * x) for x in leaves_a))
    norm_b = jnp.sqrt(sum(jnp.sum(x * x) for x in leaves_b))
    return float(dot / (norm_a * norm_b + 1e-12))


def _tree_norm(tree) -> float:
    return float(jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree.leaves(tree))))


def full_game_update(
    model,
    train_state,
    reference_params,
    ppo_config: core.PPOConfig,
    rollout_config: RolloutConfig,
    opponents: list,
    key,
    *,
    batch_size: int,
    minibatch: int,
    segment_length: int,
    gamma: float,
    gae_lambda: float,
    value_lambda: float,
):
    """opponentsの数だけ完了試合のラウンドを回し、勾配を蓄積して1回だけ更新する。

    ラウンドごとに学習側の席を入れ替える(0, 1, 0, ...)。全ラウンドのadvantageを、学習側の
    有効な行の全体で標準化する。

    Returns:
        (更新後のtrain_state, 診断値のdict)
    """
    rounds = len(opponents)
    params = train_state.params
    variables = {"params": params}
    keys = jax.random.split(key, rounds * 3)
    games = []
    for index, opponent in enumerate(opponents):
        seat = index % 2
        reset_key, run_key = keys[3 * index], keys[3 * index + 1]
        game = collect_compact_game(
            model, variables, opponent, seat, rollout_config, batch_size, reset_key, run_key
        )
        advantages, returns = game_targets(game, gamma, gae_lambda, value_lambda)
        games.append((seat, reset_key, keys[3 * index + 2], game, advantages, returns))

    valid = jnp.concatenate([g.slot_mask.any(-1).reshape(-1) for _, _, _, g, _, _ in games])
    flat_adv = jnp.concatenate([a.reshape(-1) for *_, a, _ in games])
    count = jnp.maximum(valid.sum(), 1)
    mean = jnp.sum(flat_adv * valid) / count
    std = jnp.sqrt(jnp.sum(jnp.square(flat_adv - mean) * valid) / count) + 1e-8

    round_results = []
    metric_total = core.Metrics(*(jnp.zeros(()) for _ in core.Metrics._fields))
    for seat, reset_key, permutation_key, game, advantages, returns in games:
        state = initial_state(reset_key, batch_size, rollout_config)
        steps = game.dones.shape[0]
        gradient = jax.tree.map(jnp.zeros_like, params)
        round_minibatches = 0
        for number, (start, end) in enumerate(segment_bounds(steps, segment_length)):
            window = _window(start, end)
            state, states = replay_segment(rollout_config, state, window(game.actions))
            rows = {
                "intent": window(game.intent),
                "slot_mask": game.slot_mask[start:end],
                "old_slot_log_prob": game.slot_log_prob[start:end],
                "old_value": game.value[start:end],
                "advantages": (advantages[start:end] - mean) / std,
                "returns": returns[start:end],
            }
            gradient_sum, metric_sum, used = segment_gradient(
                model,
                params,
                reference_params,
                states,
                rows,
                ppo_config,
                minibatch,
                seat,
                jax.random.fold_in(permutation_key, number),
            )
            gradient = jax.tree.map(jnp.add, gradient, gradient_sum)
            metric_total = jax.tree.map(jnp.add, metric_total, metric_sum)
            round_minibatches += int(used)
        round_results.append((gradient, round_minibatches))

    minibatches = sum(count for _, count in round_results)
    total_gradient = jax.tree.map(
        lambda *gradients: sum(gradients) / minibatches, *[g for g, _ in round_results]
    )
    diagnostics = {"gradient_norm": _tree_norm(total_gradient), "minibatches": minibatches}
    if rounds >= 2:
        per_round = [jax.tree.map(lambda x, c=c: x / c, g) for g, c in round_results[:2]]
        diagnostics["round_gradient_cosine"] = _cosine(per_round[0], per_round[1])
    new_state = train_state.apply_gradients(grads=total_gradient)
    for name in core.Metrics._fields:
        diagnostics[name] = float(getattr(metric_total, name) / minibatches)
    diagnostics.update(_game_statistics(games))
    return new_state, diagnostics


def _game_statistics(games) -> dict:
    wins = losses = draws = envs = 0
    invalid = clamped = invalid_unit = clamped_unit = daily = 0.0
    value_mean = return_mean = 0.0
    for seat, _, _, game, _, returns in games:
        margin = game.final_cash[:, seat] - game.final_cash[:, 1 - seat]
        wins += int((margin > 0).sum())
        losses += int((margin < 0).sum())
        draws += int((margin == 0).sum())
        envs += margin.shape[0]
        invalid += float(game.invalid_market.mean())
        clamped += float(game.clamped_market.mean())
        invalid_unit += float(game.invalid_unit.mean())
        clamped_unit += float(game.clamped_unit.mean())
        daily += float(
            jnp.abs(game.daily_rewards).sum() / max(int((game.daily_rewards != 0).sum()), 1)
        )
        value_mean += float(game.value.mean())
        return_mean += float(returns.mean())
    rounds = len(games)
    return {
        "games": envs,
        "win": wins / envs,
        "lose": losses / envs,
        "draw": draws / envs,
        "invalid": invalid / rounds,
        "clamped": clamped / rounds,
        "invalid_unit": invalid_unit / rounds,
        "clamped_unit": clamped_unit / rounds,
        "daily_abs_mean": daily / rounds,
        "value_mean": value_mean / rounds,
        "return_mean": return_mean / rounds,
    }
