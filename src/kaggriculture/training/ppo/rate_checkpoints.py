"""複数checkpointを総当たりさせ、勝率行列を出力するCLI。

Hydra不要の単純なスクリプト。頻繁に自動実行するものではなく、必要な時に
手動でcheckpoint群の強さを比較するための軽量ツール(Bayesian Bradley-Terryの
ような本格的な評価基盤は、今回の規模(数〜数十checkpoint)には過剰)。

使い方:
    uv run python -m kaggriculture.training.ppo.rate_checkpoints \
        outputs/ppo/baseline/2026-.../checkpoints/best \
        outputs/ppo/baseline/2026-.../checkpoints/pool/* \
        --games-per-seat 32
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import jax

from kaggriculture.policy.common.config import ModelConfig, validate_checkpoint_metadata
from kaggriculture.policy.jax import policy as P
from kaggriculture.policy.jax.model_factory import create_model
from kaggriculture.training.ppo import core
from kaggriculture.training.ppo.evaluation import evaluate_both_seats
from kaggriculture.training.ppo.rollout import RolloutConfig


def _load(directory: Path) -> tuple[dict, ModelConfig, dict]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    validate_checkpoint_metadata(metadata)
    model_config = ModelConfig(**metadata["model_config"])
    model = create_model(model_config, metadata.get("model_variant", "shared"))
    variables = P.initialize(model, jax.random.key(0))
    ppo_config = core.PPOConfig(turns_per_day=24, shed_capacity=100)
    train_state = core.create_train_state(model, variables, ppo_config)
    from kaggriculture.training.checkpoint import load_checkpoint

    restored, _ = load_checkpoint(directory, train_state)
    return {"params": restored.params}, model_config, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path, help="評価するcheckpointディレクトリ")
    parser.add_argument("--games-per-seat", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("ratings.json"))
    args = parser.parse_args()

    loaded = {}
    model = None
    variant = None
    for directory in args.checkpoints:
        variables, model_config, metadata = _load(directory)
        current_variant = metadata.get("model_variant", "shared")
        if variant is not None and current_variant != variant:
            raise ValueError("rate_checkpoints requires one model variant per invocation")
        variant = current_variant
        if model is None:
            model = create_model(model_config, variant)
        loaded[str(directory)] = variables

    rollout_config = RolloutConfig(temperature=0.8)
    key = jax.random.key(args.seed)
    matrix: dict[str, dict[str, float]] = {name: {} for name in loaded}
    wins: dict[str, list[float]] = {name: [] for name in loaded}
    for name_a, name_b in combinations(loaded, 2):
        key, matchup_key = jax.random.split(key)
        result = evaluate_both_seats(
            model,
            loaded[name_a],
            loaded[name_b],
            rollout_config,
            matchup_key,
            args.games_per_seat,
        )
        win_rate_a = float(jax.device_get(result.win_rate))
        matrix[name_a][name_b] = win_rate_a
        matrix[name_b][name_a] = 1.0 - win_rate_a
        wins[name_a].append(win_rate_a)
        wins[name_b].append(1.0 - win_rate_a)
        print(f"{name_a} vs {name_b}: {win_rate_a:.3f}")

    average = {name: sum(values) / len(values) if values else None for name, values in wins.items()}
    for name, score in sorted(average.items(), key=lambda item: -(item[1] or 0)):
        print(f"{name}: average_win_rate={score}")
    args.output.write_text(
        json.dumps({"matrix": matrix, "average_win_rate": average}, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
