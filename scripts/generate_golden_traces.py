"""kaggle-environments 公式シミュレータから、決定的な参照(ゴールデン)トレースを生成する。

GPU移植版を検証する際の正解データとして使う: 同じシードを両方のシミュレータに
流し込み、各ターンの状態を突き合わせる。
"""

import json
from pathlib import Path

from kaggle_environments import make

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "golden_traces"

# エージェントの組み合わせを変えることで、通るコードパスを変えている。
# - pass/pass: プレイヤー行動が一切無い状態。日次更新/街の消費/減衰ロジックだけを分離して見られる
# - random/random: 行動空間全体をノイジーに広くカバーする
# - starter/random: 実際のゲームロジック(植える/水やり/収穫/売却)を通す
SCENARIOS = [
    {"name": "pass_vs_pass", "agents": ["pass", "pass"], "seed": 0},
    {"name": "random_vs_random", "agents": ["random", "random"], "seed": 1},
    {"name": "starter_vs_random", "agents": ["starter", "random"], "seed": 2},
]


def run_scenario(agents, seed, episode_steps):
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": episode_steps, "seed": seed},
        debug=True,
    )
    env.run(agents)
    return env


def main(episode_steps=720):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        env = run_scenario(scenario["agents"], scenario["seed"], episode_steps)
        out_path = OUT_DIR / f"{scenario['name']}.json"
        out_path.write_text(json.dumps(env.toJSON()))

        final = env.steps[-1]
        rewards = [s.reward for s in final]
        print(f"{scenario['name']}: rewards={rewards} -> {out_path}")


if __name__ == "__main__":
    main()
