"""BC/value共同学習のbatch生成: shardをまたいで混ぜ、全サンプルを1回ずつ使う。"""

from collections import Counter
from pathlib import Path

import numpy as np

from kaggriculture.policy.jax.types import Intent
from kaggriculture.simulator.state import State
from kaggriculture.training.bc import cache
from kaggriculture.training.bc.dataset import BCBatch
from kaggriculture.training.bc.train import _spread_subset, _update_ema

SHARD_SIZE = 300


def _fake_shard(value: float) -> BCBatch:
    """価値教師が全サンプル同じ値のshard(1試合・1プレイヤー分を模す)。"""
    n = SHARD_SIZE
    return BCBatch(
        State(*(np.zeros((n,), np.float32) for _ in State._fields)),
        np.zeros((n,), np.int32),
        Intent(*(np.zeros((n,), np.int32) for _ in Intent._fields)),
        np.zeros((n, 35), bool),
        np.full((n,), value, np.float32),
    )


def _patch_shards(monkeypatch, count: int):
    shards = {Path(f"shard{index}"): _fake_shard(float(index)) for index in range(count)}
    monkeypatch.setattr(cache, "load_shard", lambda path: shards[path])
    return list(shards)


def _distinct_values_per_batch(batches) -> list[int]:
    return [len(set(batch.value_target.tolist())) for batch in batches]


def test_default_batches_come_from_one_or_two_shards(monkeypatch) -> None:
    paths = _patch_shards(monkeypatch, 4)

    batches = list(cache.iter_batches(paths, 100, seed=0, shuffle=True, drop_last=True))

    assert max(_distinct_values_per_batch(batches)) <= 2


def test_mixing_spreads_each_batch_across_shards(monkeypatch) -> None:
    paths = _patch_shards(monkeypatch, 8)

    batches = list(
        cache.iter_batches(paths, 100, seed=0, shuffle=True, drop_last=True, mix_shards=8)
    )

    assert min(_distinct_values_per_batch(batches)) >= 4


def test_every_sample_is_used_exactly_once_including_the_partial_buffer(monkeypatch) -> None:
    paths = _patch_shards(monkeypatch, 5)

    batches = list(
        cache.iter_batches(paths, 100, seed=1, shuffle=True, drop_last=False, mix_shards=4)
    )

    counts = Counter(value for batch in batches for value in batch.value_target.tolist())
    assert counts == {float(index): SHARD_SIZE for index in range(5)}


def test_validation_order_is_unchanged_without_shuffle(monkeypatch) -> None:
    paths = _patch_shards(monkeypatch, 3)

    batches = list(
        cache.iter_batches(paths, 100, seed=0, shuffle=False, drop_last=False, mix_shards=8)
    )

    assert np.concatenate([b.value_target for b in batches]).tolist() == (
        [0.0] * SHARD_SIZE + [1.0] * SHARD_SIZE + [2.0] * SHARD_SIZE
    )


def test_validation_subset_is_spread_over_the_whole_set() -> None:
    paths = [Path(f"shard{index}") for index in range(1000)]

    subset = _spread_subset(paths, batch_size=128, batches=100)

    assert len(subset) == 19
    assert subset[0] == paths[0]
    assert subset[-1] == paths[18 * (1000 // 19)]
    assert subset[-1].name != "shard18"


def test_ema_starts_at_the_first_value_and_then_smooths() -> None:
    class Metrics:
        policy_loss = 1.0
        value_loss = 3.0

    first = _update_ema(None, Metrics)
    assert first == (1.0, 3.0)

    class Later:
        policy_loss = 0.0
        value_loss = 0.0

    second = _update_ema(first, Later, decay=0.9)
    assert np.allclose(second, (0.9, 2.7))
