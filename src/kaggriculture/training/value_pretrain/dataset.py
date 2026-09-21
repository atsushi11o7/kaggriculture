"""完了済みリプレイからMonte Carlo価値教師を生成する。"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

from kaggriculture.training.replays.state import paired_observations_to_state
from kaggriculture.training.rl import estimated_assets


def _returns(
    states,
    outcome: float,
    *,
    gamma: float,
    turns_per_day: int,
    daily_reward_coefficient: float,
    daily_reward_scale: float,
    daily_reward_maximum: float,
) -> np.ndarray:
    """PPOと同じ日次資産報酬と終端勝敗から全時点のreturnを計算する。"""
    transition_count = len(states.step) - 1
    rewards = np.zeros((transition_count, 2), dtype=np.float32)
    if daily_reward_coefficient:
        assets = np.asarray(jax.device_get(estimated_assets(states)))
        margins = assets[:, 0] - assets[:, 1]
        previous_margin = margins[0]
        for index in range(transition_count):
            post_index = index + 1
            done = index == transition_count - 1
            if int(states.step[post_index]) % turns_per_day == 0 or done:
                change = margins[post_index] - previous_margin
                reward = np.clip(
                    daily_reward_coefficient * change / daily_reward_scale,
                    -daily_reward_maximum,
                    daily_reward_maximum,
                )
                rewards[index] = (reward, -reward)
                previous_margin = margins[post_index]
    rewards[-1] += (outcome, -outcome)
    returns = np.empty_like(rewards)
    running = np.zeros(2, dtype=np.float32)
    for index in range(transition_count - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def load_episode(
    path: Path,
    *,
    gamma: float,
    episode_steps: int,
    turns_per_day: int,
    daily_reward_coefficient: float = 0.0,
    daily_reward_scale: float = 10000.0,
    daily_reward_maximum: float = 0.02,
):
    """完了試合から両席の複合報酬returnを作る。

    Args:
        path: 完了済みKaggle episode JSON。
        gamma: PPOと同じ割引率。
        episode_steps: 期待するリプレイのstep数。
        turns_per_day: 1日あたりのターン数。
        daily_reward_coefficient: 日次資産差改善報酬の係数。0で終端報酬のみ。
        daily_reward_scale: 資産差改善を正規化する値。
        daily_reward_maximum: 1回の日次報酬の絶対値上限。

    Returns:
        行動前State系列と、対応する両席の割引累積報酬。

    Raises:
        ValueError: 報酬設定が不正、またはリプレイが不完全な場合。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return episode_from_data(
        data,
        path,
        gamma=gamma,
        episode_steps=episode_steps,
        turns_per_day=turns_per_day,
        daily_reward_coefficient=daily_reward_coefficient,
        daily_reward_scale=daily_reward_scale,
        daily_reward_maximum=daily_reward_maximum,
    )


def episode_from_data(
    data: dict,
    path: Path,
    *,
    gamma: float,
    episode_steps: int,
    turns_per_day: int,
    daily_reward_coefficient: float = 0.0,
    daily_reward_scale: float = 10000.0,
    daily_reward_maximum: float = 0.02,
):
    """解析済みepisodeから、行動前Stateの系列と両席のreturnを作る。

    BCとvalueを同じ局面で共同学習するとき、episodeを二重に解析しないための入口。
    返すStateの位置`i`は、`iter_replay_actions`が返す観測のstep位置`i`と一致する。
    """
    if daily_reward_coefficient < 0 or daily_reward_scale <= 0 or daily_reward_maximum <= 0:
        raise ValueError("invalid daily reward configuration")
    steps = data["steps"]
    rewards = data.get("rewards")
    if (
        len(steps) != episode_steps
        or rewards is None
        or len(rewards) != 2
        or any(not isinstance(value, (int, float)) or not np.isfinite(value) for value in rewards)
        or any(player.get("status") != "DONE" for player in steps[-1])
    ):
        raise ValueError(f"incomplete episode: {path}")
    states = []
    for index, step in enumerate(steps):
        observations = [player.get("observation") for player in step]
        if any(observation is None for observation in observations):
            raise ValueError(f"missing observation at step {index}: {path}")
        states.append(paired_observations_to_state(observations, turns_per_day=turns_per_day))
    states = jax.tree.map(lambda *fields: np.stack(fields), *states)
    outcome = float(np.sign(float(rewards[0]) - float(rewards[1])))
    targets = _returns(
        states,
        outcome,
        gamma=gamma,
        turns_per_day=turns_per_day,
        daily_reward_coefficient=daily_reward_coefficient,
        daily_reward_scale=daily_reward_scale,
        daily_reward_maximum=daily_reward_maximum,
    )
    return jax.tree.map(lambda values: values[:-1], states), targets


def iter_batches(
    paths,
    *,
    batch_size: int,
    gamma: float,
    episode_steps: int,
    turns_per_day: int,
    seed: int,
    daily_reward_coefficient: float = 0.0,
    daily_reward_scale: float = 10000.0,
    daily_reward_maximum: float = 0.02,
):
    """1試合ずつ読み込み、試合単位でシャッフルした固定shape batchを返す。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    pending_states = []
    pending_targets = []
    for path_index in order:
        try:
            states, targets = load_episode(
                paths[path_index],
                gamma=gamma,
                episode_steps=episode_steps,
                turns_per_day=turns_per_day,
                daily_reward_coefficient=daily_reward_coefficient,
                daily_reward_scale=daily_reward_scale,
                daily_reward_maximum=daily_reward_maximum,
            )
        except ValueError:
            continue
        for index in rng.permutation(len(targets)):
            pending_states.append(jax.tree.map(lambda values, index=index: values[index], states))
            pending_targets.append(targets[index])
            if len(pending_targets) == batch_size:
                yield (
                    jax.tree.map(lambda *fields: np.stack(fields), *pending_states),
                    np.stack(pending_targets),
                )
                pending_states.clear()
                pending_targets.clear()
