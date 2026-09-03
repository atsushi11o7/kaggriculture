"""kaggle-environments 公式シミュレータから、決定的な参照(ゴールデン)トレースを生成する。

GPU移植版を検証する際の正解データとして使う: 同じシードを両方のシミュレータに
流し込み、各ターンの状態を突き合わせる。
"""

import json
import sys
from pathlib import Path

from kaggle_environments import make

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage_agent import coverage_agent  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "golden_traces"

# エージェントの組み合わせを変えることで、通るコードパスを変えている。
# - pass/pass: プレイヤー行動が一切無い状態。日次更新/街の消費/減衰ロジックだけを分離して見られる
# - random/random: 行動空間全体をノイジーに広くカバーする
# - starter/random: 実際のゲームロジック(植える/水やり/収穫/売却)を通す
#
# GPU移植版はJAXの乱数(threefry)を使うため、Pythonのrandom.Random(メルセンヌ・
# ツイスタ)とはビット単位で一致しない。雑草発生・店の抽選が絡む比較は統計的検証に
# 任せ、ここでの厳密突き合わせ用シナリオ(_deterministic)は設定でその2つを無効化し、
# 残りの全ロジックを100%厳密に比較できるようにする。
DETERMINISTIC_CONFIG = {
    "weedSpawnChance": 0,  # 雑草を一切発生させない
    "townShopUnlockInterval": 100_000,  # 720ターン(30日)以内は店を一切出現させない
}

SCENARIOS = [
    {"name": "pass_vs_pass", "agents": ["pass", "pass"], "seed": 0},
    {"name": "random_vs_random", "agents": ["random", "random"], "seed": 1},
    {"name": "starter_vs_random", "agents": ["starter", "random"], "seed": 2},
    {
        "name": "random_vs_random_deterministic",
        "agents": ["random", "random"],
        "seed": 3,
        "config": DETERMINISTIC_CONFIG,
    },
    {
        "name": "starter_vs_random_deterministic",
        "agents": ["starter", "random"],
        "seed": 4,
        "config": DETERMINISTIC_CONFIG,
    },
    {
        # HIRE/BUY_LAND/BUY_ANIMAL/BUILD_*/PLACE/FEED/CARE/COLLECT_FERTILIZER/
        # FERTILIZE/DIG/PICKUP/DROPは random/starter のどちらも使わないため、
        # これらを狙って踏みに行く coverage_agent (scripts/coverage_agent.py) で
        # 別途カバレッジを取る。
        "name": "coverage_vs_random_deterministic",
        "agents": [coverage_agent, "random"],
        "seed": 5,
        "config": DETERMINISTIC_CONFIG,
        "episode_steps": 360,
    },
]


def run_scenario(agents, seed, episode_steps, extra_config=None):
    """kaggle-environmentsで1局実行し、記録済みのenvを返す。"""
    configuration = {"episodeSteps": episode_steps, "seed": seed, **(extra_config or {})}
    env = make("kaggriculture", configuration=configuration, debug=True)
    env.run(agents)
    return env


def main(episode_steps=720):
    """SCENARIOSを全て実行し、各トレースをdata/golden_traces/*.jsonへ書き出す。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        steps = scenario.get("episode_steps", episode_steps)
        env = run_scenario(scenario["agents"], scenario["seed"], steps, scenario.get("config"))
        out_path = OUT_DIR / f"{scenario['name']}.json"
        out_path.write_text(json.dumps(env.toJSON()))

        final = env.steps[-1]
        rewards = [s.reward for s in final]
        print(f"{scenario['name']}: rewards={rewards} -> {out_path}")


if __name__ == "__main__":
    main()
