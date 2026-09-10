"""実リプレイデータ(data/replays/、AGENTS.mdによりgitignore対象)を使った回帰
テスト。data/が無い環境(CI・クローン直後)ではこのファイル全体をskipする。

legal_unit_actions/legal_market_actions/decode_stateが、実際の対局ログの行動を
教師強制で再現できるか(_TeacherForceChooserが該当候補を必ず見つけられるか)を
検証する。シミュレータが黙って無視するだけの不正な要求(所持金/在庫超過)を
含むturnは失敗として扱う(evaluate_actions()のdocstring参照。既知の限界)。
"""

import json
from pathlib import Path

import pytest

from kaggriculture.policy import distribution as D

REPLAY_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "replays" / "kaggriculture-episodes-2026-08-30"
)
# このセッションで実際に検証したエピソードと、既知の合格数(6エピソード計8628ターン
# 中8540ターン一致。残りはシミュレータ側で無視されるだけの不正要求で、意図的な
# 既知の限界)。data/の内容が変わらない限りこの数値は変わらないはずで、変化したら
# legal_*_actions/decode_stateのどこかに退行が入ったことを意味する。
EXPECTED_EPISODES = {
    "103268405.json": 1438,
    "103663271.json": 1437,
    "102866485.json": 1438,
    "102866504.json": 1385,
    "102879960.json": 1404,
    "102891139.json": 1438,
}
EXPECTED_TOTAL = 8540

pytestmark = pytest.mark.skipif(
    not REPLAY_DIR.exists(), reason="data/replays (gitignore対象) がこの環境に存在しない"
)


def _iter_pairs(episode_path: Path):
    with open(episode_path) as f:
        data = json.load(f)
    steps = data["steps"]
    n_players = len(steps[0])
    for step_idx in range(1, len(steps)):
        for p in range(n_players):
            obs = steps[step_idx - 1][p]["observation"]
            action = steps[step_idx][p]["action"]
            if action is not None:
                yield obs, action


def test_legal_action_regression(net):
    total_ok = 0
    per_episode_ok = {}
    for filename in EXPECTED_EPISODES:
        path = REPLAY_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} が見つからない")
        ok = 0
        for obs, action in _iter_pairs(path):
            try:
                D.evaluate_actions(net, obs, action)
                ok += 1
            except ValueError:
                pass
        per_episode_ok[filename] = ok
        total_ok += ok

    assert per_episode_ok == EXPECTED_EPISODES
    assert total_ok == EXPECTED_TOTAL
