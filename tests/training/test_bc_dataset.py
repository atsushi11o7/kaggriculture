"""training.bc.datasetの契約テスト。

合成observation(_fresh_observation。tests/policy/conftest.pyのmake_fresh_observation
と同じ形だが、trainingはpolicy側のテストfixtureに依存しないよう最小限だけ複製する)
で完結する部分と、実リプレイデータ(data/replays/、gitignore対象)に依存する回帰
テストに分かれる。後者はdata/が無い環境(CI・クローン直後)ではskipする。
"""

import csv
import json
from pathlib import Path

import pytest

from kaggriculture.rules import constants as C
from kaggriculture.training.bc.dataset import ReplayActionDataset
from kaggriculture.training.replays import (
    filter_episodes_by_agent_score,
    iter_replay_samples,
    list_episode_files,
    load_rating_manifest,
    split_episode_files,
)

REPLAY_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "replays" / "kaggriculture-episodes-2026-08-30"
)

BOARD_SIZE = 10


def _fresh_observation(player: int = 0) -> dict:
    """day 0(reset.py相当)のobservationを1人称分組み立てる。tests/policy/conftest.py
    のmake_fresh_observationと同じ形(BC datasetはpolicyのfixtureに依存しないため、
    ここでは最小限だけ複製する)。"""
    half = BOARD_SIZE // 2
    tiles = [
        [None if y < half and x < half else "LOCKED" for x in range(BOARD_SIZE)]
        for y in range(BOARD_SIZE)
    ]
    farm = {
        "farmer": [half - 1, half - 1],
        "hands": [],
        "hires_today": 0,
        "money": 3000.0,
        "tiles": tiles,
        "unlocked_quadrants": ["NW"],
    }
    return {
        "day": 0,
        "hour": 0,
        "step": 0,
        "player": player,
        "farms": [farm, farm],
        "private": {
            "inventories": [{}],
            "seeds": dict.fromkeys(C.CROPS, 0),
            "shed": dict.fromkeys(C.SHED_ITEMS, 0),
        },
        "market": {
            "inventory": dict.fromkeys(C.PRODUCTS, 10_000),
            "prices": dict.fromkeys(C.PRODUCTS, 100),
        },
        "town": {"unlocked_shops": []},
        "remainingOverageTime": 60,
    }


def _write_episode(path: Path, steps: list) -> None:
    path.write_text(json.dumps({"steps": steps}))


def test_list_episode_files_sorts_and_limits(tmp_path):
    for name in ["b.json", "a.json", "c.json"]:
        (tmp_path / name).write_text("{}")

    files = list_episode_files(tmp_path)
    assert [f.name for f in files] == ["a.json", "b.json", "c.json"]

    limited = list_episode_files(tmp_path, num_episodes=2)
    assert [f.name for f in limited] == ["a.json", "b.json"]


def test_split_episode_files_is_deterministic_and_covers_all(tmp_path):
    files = [tmp_path / f"{i}.json" for i in range(10)]
    train_a, val_a = split_episode_files(files, val_fraction=0.3, seed=0)
    train_b, val_b = split_episode_files(files, val_fraction=0.3, seed=0)

    assert train_a == train_b
    assert val_a == val_b
    assert len(val_a) == 3
    assert set(train_a) | set(val_a) == set(files)
    assert set(train_a).isdisjoint(val_a)


def test_split_episode_files_keeps_nonempty_validation_for_small_dataset(tmp_path):
    files = [tmp_path / f"{i}.json" for i in range(3)]

    train, val = split_episode_files(files, val_fraction=0.1, seed=0)

    assert len(train) == 2
    assert len(val) == 1


def test_split_episode_files_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        split_episode_files([], val_fraction=1.0, seed=0)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "episode_id",
        "create_time",
        "avg_score",
        "min_score",
        "sum_score",
        "agent_count",
        "size_bytes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_manifest_merges_multiple_csv_files(tmp_path):
    _write_manifest(
        tmp_path / "2026-08-30.csv",
        [
            {
                "episode_id": "1",
                "create_time": "t",
                "avg_score": "100",
                "min_score": "90",
                "sum_score": "200",
                "agent_count": "2",
                "size_bytes": "1",
            }
        ],
    )
    _write_manifest(
        tmp_path / "2026-08-31.csv",
        [
            {
                "episode_id": "2",
                "create_time": "t",
                "avg_score": "50",
                "min_score": "40",
                "sum_score": "100",
                "agent_count": "2",
                "size_bytes": "1",
            }
        ],
    )

    manifest = load_rating_manifest(tmp_path)

    assert set(manifest.keys()) == {"1", "2"}
    assert manifest["1"]["avg_score"] == "100"


def test_filter_episodes_by_agent_score_excludes_low_score_and_unlisted_episodes(tmp_path):
    _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "episode_id": "high",
                "create_time": "t",
                "avg_score": "3000",
                "min_score": "2900",
                "sum_score": "5900",
                "agent_count": "2",
                "size_bytes": "1",
            },
            {
                "episode_id": "low",
                "create_time": "t",
                "avg_score": "100",
                "min_score": "90",
                "sum_score": "190",
                "agent_count": "2",
                "size_bytes": "1",
            },
        ],
    )
    manifest = load_rating_manifest(tmp_path)
    files = [tmp_path / "high.json", tmp_path / "low.json", tmp_path / "unknown.json"]

    kept = filter_episodes_by_agent_score(files, manifest, min_avg_agent_score=1000)

    assert kept == [tmp_path / "high.json"]


def test_filter_episodes_by_agent_score_uses_worst_player_score(tmp_path):
    _write_manifest(
        tmp_path / "manifest.csv",
        [
            {
                "episode_id": "balanced",
                "create_time": "t",
                "avg_score": "3000",
                "min_score": "2900",
                "sum_score": "6000",
                "agent_count": "2",
                "size_bytes": "1",
            },
            {
                "episode_id": "unbalanced",
                "create_time": "t",
                "avg_score": "3000",
                "min_score": "100",
                "sum_score": "6000",
                "agent_count": "2",
                "size_bytes": "1",
            },
        ],
    )
    manifest = load_rating_manifest(tmp_path)
    files = [tmp_path / "balanced.json", tmp_path / "unbalanced.json"]

    kept = filter_episodes_by_agent_score(files, manifest, min_agent_score=1000)

    assert kept == [tmp_path / "balanced.json"]


def test_iter_replay_samples_yields_both_players_skipping_none_actions(tmp_path):
    obs0 = _fresh_observation(player=0)
    obs1 = _fresh_observation(player=1)
    action_pass = {"farmer": ["PASS"], "hands": [], "market": []}
    steps = [
        [{"observation": obs0, "action": None}, {"observation": obs1, "action": None}],
        [{"observation": obs0, "action": action_pass}, {"observation": obs1, "action": None}],
    ]
    path = tmp_path / "ep.json"
    _write_episode(path, steps)

    pairs = list(iter_replay_samples(path))

    assert len(pairs) == 1
    obs, action = pairs[0]
    assert obs == obs0
    assert action == action_pass


def test_iter_replay_samples_filters_selected_player(tmp_path):
    obs0 = _fresh_observation(player=0)
    obs1 = _fresh_observation(player=1)
    action_pass = {"farmer": ["PASS"], "hands": [], "market": []}
    path = tmp_path / "ep.json"
    path.write_text(
        json.dumps(
            {
                "steps": [
                    [
                        {"observation": obs0, "action": None},
                        {"observation": obs1, "action": None},
                    ],
                    [
                        {"observation": obs0, "action": action_pass},
                        {"observation": obs1, "action": action_pass},
                    ],
                ]
            }
        )
    )

    pairs = list(iter_replay_samples(path, selected_players={1}))

    assert len(pairs) == 1
    assert pairs[0][0]["player"] == 1


def test_iter_replay_samples_filters_each_player_by_terminal_reward(tmp_path):
    obs0 = _fresh_observation(player=0)
    obs1 = _fresh_observation(player=1)
    action_pass = {"farmer": ["PASS"], "hands": [], "market": []}
    path = tmp_path / "ep.json"
    path.write_text(
        json.dumps(
            {
                "rewards": [80_000, 40_000],
                "steps": [
                    [
                        {"observation": obs0, "action": None},
                        {"observation": obs1, "action": None},
                    ],
                    [
                        {"observation": obs0, "action": action_pass},
                        {"observation": obs1, "action": action_pass},
                    ],
                ],
            }
        )
    )

    pairs = list(iter_replay_samples(path, min_player_reward=60_000))

    assert len(pairs) == 1
    assert pairs[0][0]["player"] == 0


def test_replay_action_dataset_normalizes_invalid_actions(tmp_path):
    """納屋が空のfresh_obsに対しSELL WHEATを要求する無効なexpert行動が、
    MARKET_WAITへ正規化された上でyieldされる(削除ではなく置き換え)。"""
    obs = _fresh_observation(player=0)
    invalid_action = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 1]]}
    steps = [
        [{"observation": obs, "action": None}],
        [{"observation": obs, "action": invalid_action}],
    ]
    path = tmp_path / "ep.json"
    _write_episode(path, steps)

    dataset = ReplayActionDataset([path])
    samples = list(dataset)

    assert len(samples) == 1
    _, normalized = samples[0]
    assert normalized["market"][0] != ["SELL", "WHEAT", 1]


def test_replay_action_dataset_reuses_normalized_cache(tmp_path, monkeypatch):
    obs = _fresh_observation(player=0)
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    episode = tmp_path / "ep.json"
    _write_episode(
        episode,
        [
            [{"observation": obs, "action": None}],
            [{"observation": obs, "action": action}],
        ],
    )
    cache_dir = tmp_path / "cache"
    expected = list(ReplayActionDataset([episode], cache_dir=cache_dir))
    assert len(list(cache_dir.glob("*.jsonl.gz"))) == 1

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("normalization should be loaded from cache")

    monkeypatch.setattr(
        "kaggriculture.training.bc.dataset.D.normalize_expert_action", fail_if_recomputed
    )
    actual = list(ReplayActionDataset([episode], cache_dir=cache_dir))
    assert actual == expected


def test_replay_action_dataset_shards_episodes_across_workers(tmp_path, monkeypatch):
    import torch.utils.data as torch_data

    obs = _fresh_observation(player=0)
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    files = []
    for i in range(4):
        path = tmp_path / f"ep{i}.json"
        _write_episode(
            path,
            [
                [{"observation": obs, "action": None}],
                [{"observation": obs, "action": action}],
            ],
        )
        files.append(path)

    ds = ReplayActionDataset(files)

    class FakeWorkerInfo:
        id = 1
        num_workers = 2

    monkeypatch.setattr(torch_data, "get_worker_info", lambda: FakeWorkerInfo())
    assert ds._files_for_this_worker() == [files[1], files[3]]


@pytest.mark.skipif(
    not REPLAY_DIR.exists(), reason="data/replays (gitignore対象) がこの環境に存在しない"
)
def test_replay_action_dataset_smoke_on_real_data():
    files = list_episode_files(REPLAY_DIR, num_episodes=1)
    assert files
    dataset = ReplayActionDataset(files)
    samples = list(dataset)
    assert len(samples) > 0
    for obs, action in samples:
        assert isinstance(obs, dict)
        assert isinstance(action, dict)
