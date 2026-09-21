"""Critic事前学習用のepisode shard cacheとbatch iterator。

生JSONの読み込み・(state, 割引済みreturn)への変換は試合ごとに重いため、BCの
cache.pyと同じ方式で1試合を固定shape shardへ一度だけ変換して保存する。
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from kaggriculture.simulator.state import State
from kaggriculture.training.value_pretrain.dataset import load_episode

_VERSION = 1
_STATE_PREFIX = "state__"


def _path(
    source: Path,
    directory: Path,
    *,
    gamma: float,
    episode_steps: int,
    turns_per_day: int,
    daily_reward_coefficient: float,
    daily_reward_scale: float,
    daily_reward_maximum: float,
) -> Path:
    stat = source.stat()
    identity = (
        str(source.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        gamma,
        episode_steps,
        turns_per_day,
        daily_reward_coefficient,
        daily_reward_scale,
        daily_reward_maximum,
        _VERSION,
    )
    return directory / f"{hashlib.sha256(repr(identity).encode()).hexdigest()}.npz"


def prepare_episode(
    source: Path,
    directory: Path,
    *,
    gamma: float,
    episode_steps: int,
    turns_per_day: int,
    daily_reward_coefficient: float = 0.0,
    daily_reward_scale: float = 10000.0,
    daily_reward_maximum: float = 0.02,
) -> Path | None:
    """1 episodeを(state, target)固定shape shardへatomicに保存する。"""
    destination = _path(
        source,
        directory,
        gamma=gamma,
        episode_steps=episode_steps,
        turns_per_day=turns_per_day,
        daily_reward_coefficient=daily_reward_coefficient,
        daily_reward_scale=daily_reward_scale,
        daily_reward_maximum=daily_reward_maximum,
    )
    if destination.exists():
        return destination
    try:
        states, targets = load_episode(
            source,
            gamma=gamma,
            episode_steps=episode_steps,
            turns_per_day=turns_per_day,
            daily_reward_coefficient=daily_reward_coefficient,
            daily_reward_scale=daily_reward_scale,
            daily_reward_maximum=daily_reward_maximum,
        )
    except ValueError:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{_STATE_PREFIX}{name}": value for name, value in zip(State._fields, states, strict=True)
    }
    arrays["targets"] = targets
    temporary = destination.with_suffix(f".tmp-{os.getpid()}")
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def prepare_episodes(sources, directory: Path, **reward_kwargs) -> list[Path]:
    """episode群を変換し、利用可能なshard pathを返す。"""
    paths = []
    for source in sources:
        result = prepare_episode(Path(source), directory, **reward_kwargs)
        if result is not None:
            paths.append(result)
    return paths


def load_shard(path: Path) -> tuple[State, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        states = State(*(data[f"{_STATE_PREFIX}{name}"] for name in State._fields))
        return states, data["targets"]


def _take(states: State, targets: np.ndarray, index) -> tuple[State, np.ndarray]:
    return State(*(value[index] for value in states)), targets[index]


def _concat(
    left: tuple[State, np.ndarray] | None, right: tuple[State, np.ndarray]
) -> tuple[State, np.ndarray]:
    if left is None:
        return right

    def join(a, b):
        return np.concatenate([a, b])

    left_states, left_targets = left
    right_states, right_targets = right
    return (
        State(*(join(a, b) for a, b in zip(left_states, right_states, strict=True))),
        join(left_targets, right_targets),
    )


def _prefetch(paths: Sequence[Path]) -> Iterator[tuple[State, np.ndarray]]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = None
        for path in paths:
            following = executor.submit(load_shard, path)
            if future is not None:
                yield future.result()
            future = following
        if future is not None:
            yield future.result()


def iter_batches(
    paths, batch_size: int, *, seed: int, shuffle: bool = True, drop_last: bool = True
):
    """shardを1つ先読みし、全データをRAMへ置かず固定batch化する。"""
    rng = np.random.default_rng(seed)
    ordered = list(paths)
    if shuffle:
        rng.shuffle(ordered)
    pending = None
    for shard in _prefetch(ordered):
        if shuffle:
            states, targets = shard
            shard = _take(states, targets, rng.permutation(len(targets)))
        pending = _concat(pending, shard)
        while len(pending[1]) >= batch_size:
            yield _take(*pending, slice(0, batch_size))
            pending = _take(*pending, slice(batch_size, None))
    if pending is not None and len(pending[1]) and not drop_last:
        yield pending
