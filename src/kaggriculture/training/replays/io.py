"""Kaggle episode JSONの列挙・分割・サンプル読込。"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path


def list_episode_files(data_dir: Path, num_episodes: int | None = None) -> list[Path]:
    """data_dir以下のepisode JSONを安定した順序で列挙する。"""
    files = sorted(data_dir.glob("**/*.json"))
    return files if num_episodes is None else files[:num_episodes]


def split_episode_files(
    files: list[Path], val_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """episode単位で再現可能なtrain/validation分割を作る。"""
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    count = int(len(shuffled) * val_fraction)
    if val_fraction > 0 and len(shuffled) > 1:
        count = max(1, count)
    return shuffled[count:], shuffled[:count]


def iter_replay_samples(
    episode_path: Path,
    min_player_reward: float | None = None,
    selected_players: set[int] | None = None,
) -> Iterator[tuple[dict, dict]]:
    """1 episodeから指定プレイヤーの(観測, 行動)を時系列に列挙する。"""
    with open(episode_path, encoding="utf-8") as stream:
        data = json.load(stream)
    steps = data["steps"]
    player_count = len(steps[0])
    rewards = data.get("rewards")
    if min_player_reward is not None and (
        not isinstance(rewards, list) or len(rewards) != player_count
    ):
        raise ValueError(f"missing player rewards in {episode_path}")
    eligible = [
        player
        for player in range(player_count)
        if (selected_players is None or player in selected_players)
        and (min_player_reward is None or float(rewards[player]) >= min_player_reward)
    ]
    for step_index in range(1, len(steps)):
        for player in eligible:
            observation = steps[step_index - 1][player]["observation"]
            action = steps[step_index][player]["action"]
            if action is not None:
                yield observation, action
