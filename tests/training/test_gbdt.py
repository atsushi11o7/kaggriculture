"""GBDT教師の特徴量・ランキング・自己回帰接続テスト。"""

import json
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from kaggriculture.policy.torch.candidate_api import decode_with_candidates
from kaggriculture.training.gbdt.agent import GBDTAgent
from kaggriculture.training.gbdt.dataset import (
    BuildStats,
    RankingData,
    build_ranking_data,
    load_ranking_data,
    load_ranking_stats,
    save_ranking_data,
)
from kaggriculture.training.gbdt.features import FEATURE_NAMES, context_features, group_features
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
    assert train.data.num_episodes > 0
    assert generate.environment.maxMarketOrdersPerTurn == train.rules.max_market_orders
