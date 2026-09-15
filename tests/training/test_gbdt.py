"""GBDT教師の特徴量・ランキング・自己回帰接続テスト。"""

import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from kaggriculture.policy.torch.candidate_api import decode_with_candidates
from kaggriculture.training.gbdt.agent import GBDTAgent
from kaggriculture.training.gbdt.dataset import (
    BuildStats,
    RankingData,
    _balanced_class_weights,
    _quantity_indices,
    build_ranking_data,
    build_ranking_files,
    load_ranking_data,
    load_ranking_files,
    load_ranking_stats,
    open_ranking_data,
    save_ranking_data,
)
from kaggriculture.training.gbdt.features import (
    FEATURE_NAMES,
    classify_market_op_row,
    context_features,
    group_features,
)
from kaggriculture.training.gbdt.model import GBDTRanker, RankerConfig
from tests.policy.conftest import make_fresh_observation

_CONFIG_DIR = Path(__file__).parents[2] / "src/kaggriculture/training/conf"


def _episode(path: Path) -> Path:
    obs = make_fresh_observation(0)
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    path.write_text(
        json.dumps(
            {
                "rewards": [1.0],
                "steps": [
                    [{"observation": obs, "action": None}],
                    [{"observation": obs, "action": action}],
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_replay_builds_grouped_candidate_rows(tmp_path: Path) -> None:
    stats = BuildStats()
    data = build_ranking_data([_episode(tmp_path / "episode.json")], stats=stats)
    assert set(data) == {"unit_op", "market_op"}
    for group in data.values():
        assert group.features.shape[1] == len(FEATURE_NAMES)
        assert group.features.shape[0] == group.labels.shape[0] == group.groups.sum()
        assert group.labels.sum() == len(group.groups)
    assert stats.accepted == 1
    assert stats.discarded == 0

    cache = tmp_path / "ranking.npz"
    save_ranking_data(cache, data, stats)
    loaded = load_ranking_data(cache)
    np.testing.assert_array_equal(loaded["unit_op"].groups, data["unit_op"].groups)
    assert load_ranking_stats(cache).to_dict() == stats.to_dict()


def _episode_with_market_action(path: Path, market: list) -> Path:
    obs = make_fresh_observation(0)
    action = {"farmer": ["PASS"], "hands": [], "market": market}
    path.write_text(
        json.dumps(
            {
                "rewards": [1.0],
                "steps": [
                    [{"observation": obs, "action": None}],
                    [{"observation": obs, "action": action}],
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_balanced_class_weights_upweights_rare_classes_and_normalizes_to_one() -> None:
    # クラス0が9割、クラス1が1割を占める偏った分布。
    codes = [0] * 90 + [1] * 10
    weights = _balanced_class_weights(codes, num_classes=2)

    assert weights[1] > weights[0]
    counts = np.bincount(codes, minlength=2)
    weighted_mean = float(np.sum(weights * counts) / counts.sum())
    assert weighted_mean == pytest.approx(1.0)


def test_balanced_class_weights_clips_extreme_imbalance() -> None:
    # 1件しかないクラスへの完全な逆頻度は極端になるが、平方根+クリップで抑える。
    codes = [0] * 999 + [1] * 1
    weights = _balanced_class_weights(codes, num_classes=2)

    # 正規化前のクリップ上限は8.0、正規化後もクラス0側との比は大きすぎない。
    assert weights[1] / weights[0] < 8.0 * 4  # 完全な逆頻度(999倍)よりずっと小さい


def test_build_ranking_files_upweights_rare_market_op_class(tmp_path: Path) -> None:
    # STOPだけの局面を多数、BUY_SEEDを選ぶ局面を少数混ぜ、rareクラスの重みが
    # 高くなること、classify_market_op_rowで復元したクラスと一致することを確認する。
    episodes = [_episode_with_market_action(tmp_path / f"stop_{i}.json", []) for i in range(9)]
    episodes.append(
        _episode_with_market_action(tmp_path / "buy_seed.json", [["BUY_SEED", "WHEAT", 1]])
    )

    files = build_ranking_files(episodes, tmp_path / "ranking", quantity_limit=24)
    data = open_ranking_data(files, "market_op")

    assert data.weights is not None
    # 重みはquery(決定)単位で一様なので、各groupの「実際に選ばれた候補」の
    # クラスを見て、そのgroupの重みと突き合わせる(候補行を無差別に集計すると
    # 同じgroup内の非選択候補まで混じって意味が薄れる)。
    offset = 0
    weight_by_selected_class: dict[str, float] = {}
    for size in data.groups:
        size = int(size)
        rows = data.features[offset : offset + size]
        labels = data.labels[offset : offset + size]
        weights = data.weights[offset : offset + size]
        selected = int(np.argmax(labels))
        selected_class = classify_market_op_row(rows[selected])
        weight_by_selected_class[selected_class] = float(weights[0])
        assert np.all(weights == weights[0])  # group内は一様な重み
        offset += size

    assert weight_by_selected_class["BUY_SEED"] > weight_by_selected_class["STOP"]


def _market_op_row(op_name: str) -> np.ndarray:
    row = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    if op_name == "WAIT":
        row[FEATURE_NAMES.index("market_wait")] = 1
    elif op_name == "STOP":
        row[FEATURE_NAMES.index("market_stop")] = 1
    else:
        row[FEATURE_NAMES.index(f"market_op_{op_name}")] = 1
    return row


def test_class_top1_report_distinguishes_op_level_from_exact_match() -> None:
    # class_top1_reportの「exact」(候補indexの完全一致、作物まで一致必須)と
    # 「op」(op名だけの一致)は別物であることを固定する回帰テスト。1 groupに
    # BUY_SEEDの候補を2つ(作物は区別しない)置き、正解はindex 1、modelは
    # 未学習(models={})でスコアが常にゼロのためargmaxは常にindex 0を予測する。
    # op同士は一致するが、候補indexはずれる。
    features = np.stack([_market_op_row("BUY_SEED"), _market_op_row("BUY_SEED")])
    labels = np.array([0, 1], dtype=np.int8)
    groups = np.array([2], dtype=np.int32)
    data = RankingData(features, labels, groups)

    ranker = GBDTRanker({}, RankerConfig())
    report = ranker.class_top1_report("market_op", data)

    assert report["per_class_op_accuracy"]["BUY_SEED"] == 1.0
    assert report["per_class_accuracy"]["BUY_SEED"] == 0.0


def test_quantity_sampling_preserves_expert_and_boundaries() -> None:
    indices = _quantity_indices(100, selected=72, limit=24)

    assert len(indices) == 24
    assert indices.tolist() == sorted(set(indices.tolist()))
    assert {0, 72, 99}.issubset(indices)
    assert {70, 71, 73, 74}.issubset(indices)


def test_disk_backed_ranking_data_round_trip(tmp_path: Path) -> None:
    files = build_ranking_files(
        [_episode(tmp_path / "episode.json")], tmp_path / "ranking", quantity_limit=24
    )
    loaded = load_ranking_files(tmp_path / "ranking")

    assert loaded.stats.accepted == files.stats.accepted == 1
    unit = open_ranking_data(loaded, "unit_op")
    assert isinstance(unit.features, np.memmap)
    assert unit.features.shape[0] == unit.labels.shape[0] == unit.groups.sum()
    assert unit.features.shape[1] == len(FEATURE_NAMES)
    ranker = GBDTRanker.fit(loaded, RankerConfig(n_estimators=2, min_child_samples=1, n_jobs=1))
    assert set(ranker.models) == {"unit_op", "market_op"}


def test_ranker_round_trip_and_agent_action(tmp_path: Path) -> None:
    obs = make_fresh_observation(0)
    captured = {}

    def capture(decision):
        if not captured:
            captured["decision"] = decision
        return 0

    decode_with_candidates(obs, capture)
    decision = captured["decision"]
    kind = decision.kind
    context = context_features(
        obs, decision.context, kind, decision.position, decision.board_position
    )
    features = group_features(
        context, decision.candidates, decision.context, decision.board_position
    )
    matrix = np.concatenate([features, features, features])
    labels = np.zeros(len(matrix), dtype=np.int8)
    # 任意の合法候補を正例にし、rankerの保存・復元とagent接続を検証する。
    selected_index = 0
    labels[
        [
            selected_index,
            len(decision.candidates) + selected_index,
            2 * len(decision.candidates) + selected_index,
        ]
    ] = 1
    data = {
        kind: RankingData(
            features=matrix, labels=labels, groups=np.full(3, len(decision.candidates))
        )
    }
    ranker = GBDTRanker.fit(data, RankerConfig(n_estimators=2, min_child_samples=1, n_jobs=1))
    checkpoint = tmp_path / "model"
    ranker.save(checkpoint)
    loaded = GBDTRanker.load(checkpoint)
    np.testing.assert_allclose(loaded.score(kind, features), ranker.score(kind, features))

    action = GBDTAgent(loaded).act(obs)
    assert set(action) == {"farmer", "hands", "market"}


def test_gbdt_hydra_configs_compose() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR.resolve())):
        train = compose(config_name="gbdt")
        generate = compose(config_name="gbdt_generate", overrides=["model.checkpoint=/tmp/teacher"])
    assert train.data.num_episodes is None
    assert train.data.quantity_candidates == 16
    assert generate.environment.maxMarketOrdersPerTurn == train.rules.max_market_orders
