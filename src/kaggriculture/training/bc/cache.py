"""固定slot BC用のepisode shard cacheとbatch iterator。"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from kaggriculture.policy.jax.types import Intent
from kaggriculture.simulator.state import State
from kaggriculture.training.bc.dataset import (
    BCBatch,
    BuildStats,
    iter_samples,
    stack_samples,
)
from kaggriculture.training.replays.state import CacheRules

_VERSION = 3
_STATE_PREFIX = "state__"
_INTENT_PREFIX = "intent__"


def _path(source: Path, directory: Path, rules: CacheRules, players, reward=None) -> Path:
    """試合の内容(名前・サイズ・更新時刻)と設定で決まる。同じ試合を別ディレクトリへ
    置き直しても(ハードリンクなど)、キャッシュを再利用できる。"""
    stat = source.stat()
    reward_key = None if reward is None else tuple(sorted(reward.items()))
    identity = (source.name, stat.st_size, stat.st_mtime_ns, rules, players, reward_key, _VERSION)
    return directory / f"{hashlib.sha256(repr(identity).encode()).hexdigest()}.npz"


def prepare_episode(
    source: Path, directory: Path, rules: CacheRules, players=None, reward: dict | None = None
) -> Path | None:
    """1 episodeを固定shape shardへatomicに保存する。

    rewardを渡すと、同じ(試合・step・プレイヤー)の価値教師も同じshardへ入れる。
    """
    destination = _path(source, directory, rules, players, reward)
    if destination.exists():
        return destination
    stats = BuildStats()
    selected = (source, players) if players is not None else source
    samples = list(iter_samples([selected], rules, stats, reward))
    directory.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".stats.json").write_text(
        json.dumps(
            {
                "accepted": stats.accepted,
                "discarded": stats.discarded,
                "reasons": dict(stats.reasons),
            },
            sort_keys=True,
        )
    )
    if not samples:
        return None
    batch = stack_samples(samples)
    arrays = {
        **{
            f"{_STATE_PREFIX}{name}": value
            for name, value in zip(State._fields, batch.states, strict=True)
        },
        **{
            f"{_INTENT_PREFIX}{name}": value
            for name, value in zip(Intent._fields, batch.intent, strict=True)
        },
        "players": batch.players,
        "slot_mask": batch.slot_mask,
        "value_target": batch.value_target,
    }
    temporary = destination.with_suffix(f".tmp-{os.getpid()}")
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def prepare_episodes(sources, directory: Path, rules: CacheRules, reward: dict | None = None):
    """episode群を変換し、利用可能なshard pathを返す。"""
    paths = []
    for item in sources:
        source, players = item if isinstance(item, tuple) else (item, None)
        result = prepare_episode(Path(source), directory, rules, players, reward)
        if result is not None:
            paths.append(result)
    return paths


def load_shard(path: Path) -> BCBatch:
    with np.load(path, allow_pickle=False) as data:
        states = State(*(data[f"{_STATE_PREFIX}{name}"] for name in State._fields))
        intent = Intent(*(data[f"{_INTENT_PREFIX}{name}"] for name in Intent._fields))
        return BCBatch(states, data["players"], intent, data["slot_mask"], data["value_target"])


def _take(batch: BCBatch, index) -> BCBatch:
    return BCBatch(
        State(*(value[index] for value in batch.states)),
        batch.players[index],
        Intent(*(value[index] for value in batch.intent)),
        batch.slot_mask[index],
        batch.value_target[index],
    )


def _concat(left: BCBatch | None, right: BCBatch) -> BCBatch:
    if left is None:
        return right

    def join(a, b):
        return np.concatenate([a, b])

    return BCBatch(
        State(*(join(a, b) for a, b in zip(left.states, right.states, strict=True))),
        join(left.players, right.players),
        Intent(*(join(a, b) for a, b in zip(left.intent, right.intent, strict=True))),
        join(left.slot_mask, right.slot_mask),
        join(left.value_target, right.value_target),
    )


def _prefetch(paths: Sequence[Path]) -> Iterator[BCBatch]:
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
    paths,
    batch_size: int,
    *,
    seed: int,
    shuffle: bool,
    drop_last: bool,
    mix_shards: int = 1,
):
    """shardを1つ先読みし、全データをRAMへ置かず固定batch化する。

    1 shardは1試合・1プレイヤー分で、内部の教師(特にvalueの符号)が偏る。mix_shardsが2以上なら、
    その数のshardをまとめて全サンプルを混ぜてからbatch化し、1 batchが複数の試合・席・勝敗を含む
    ようにする(shuffle時のみ)。
    """
    rng = np.random.default_rng(seed)
    ordered = list(paths)
    if shuffle:
        rng.shuffle(ordered)
    mix = max(mix_shards, 1) if shuffle else 1
    pending = None
    loaded = 0
    for shard in _prefetch(ordered):
        if shuffle and mix == 1:
            shard = _take(shard, rng.permutation(len(shard.players)))
        pending = _concat(pending, shard)
        loaded += 1
        if mix > 1:
            if loaded % mix:
                continue
            pending = _take(pending, rng.permutation(len(pending.players)))
        while len(pending.players) >= batch_size:
            yield _take(pending, slice(0, batch_size))
            pending = _take(pending, slice(batch_size, None))
    if pending is not None and mix > 1 and loaded % mix:
        pending = _take(pending, rng.permutation(len(pending.players)))
        while len(pending.players) >= batch_size:
            yield _take(pending, slice(0, batch_size))
            pending = _take(pending, slice(batch_size, None))
    if pending is not None and len(pending.players) and not drop_last:
        yield pending
