"""固定seedのペア評価で、複数のcheckpointを総当たりで比較するCLI。

各対局は同じ盤面seedで座席を入れ替えて2回行い、盤面・座席の偏りを打ち消す。
全ての組み合わせが同じseedを使うため、checkpoint間の差に環境乱数の差が混ざらない。

例:
  uv run python -m kaggriculture.training.ppo.league \\
    --candidate step20=ppo:outputs/ppo/run/checkpoints/step_20 \\
    --opponent anchor=bc:outputs/bc/recent5/.../checkpoints/best \\
    --opponent best=ppo:outputs/ppo/run/checkpoints/best
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import policy as P
from kaggriculture.policy.jax.model_factory import create_model
from kaggriculture.training.checkpoint import read_checkpoint_metadata
from kaggriculture.training.ppo import core
from kaggriculture.training.ppo.evaluation import evaluate_closed_loop
from kaggriculture.training.ppo.rollout import RolloutConfig
from kaggriculture.training.ppo.train import _load_actor_checkpoint, _opponent_variables


def wilson_interval(score: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """勝率(勝ち=1・引き分け=0.5)のWilson信頼区間。引き分けは近似として扱う。"""
    if games == 0:
        return float("nan"), float("nan")
    denominator = 1 + z**2 / games
    center = (score + z**2 / (2 * games)) / denominator
    half = z * math.sqrt(score * (1 - score) / games + z**2 / (4 * games**2)) / denominator
    return center - half, center + half


def _score(margin: np.ndarray) -> float:
    return float(np.mean(np.where(margin > 0, 1.0, np.where(margin < 0, 0.0, 0.5))))


def play_paired(
    model,
    candidate: dict,
    opponent: dict,
    config: RolloutConfig,
    seed_key: jax.Array,
    games_per_seat: int,
    greedy: bool = True,
) -> dict:
    """同じ盤面seedで、候補をseat0・seat1の両方に置いて対局し、候補視点の結果を返す。"""
    as_seat0 = evaluate_closed_loop(
        model, candidate, opponent, config, seed_key, games_per_seat, greedy=greedy
    )
    as_seat1 = evaluate_closed_loop(
        model, opponent, candidate, config, seed_key, games_per_seat, greedy=greedy
    )
    margin = np.concatenate(
        [
            np.asarray(as_seat0.cash[:, 0] - as_seat0.cash[:, 1]),
            np.asarray(as_seat1.cash[:, 1] - as_seat1.cash[:, 0]),
        ]
    )
    wins = int((margin > 0).sum())
    losses = int((margin < 0).sum())
    draws = int((margin == 0).sum())
    games = len(margin)
    score = (wins + 0.5 * draws) / games
    low, high = wilson_interval(score, games)
    pass_rate = np.concatenate(
        [np.asarray(as_seat0.pass_rate), np.asarray(as_seat1.opponent_pass_rate)]
    )
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "ci_low": low,
        "ci_high": high,
        "seat0_score": _score(margin[: games // 2]),
        "seat1_score": _score(margin[games // 2 :]),
        "margin_mean": float(margin.mean()),
        "margin_median": float(np.median(margin)),
        "pass_rate": float(pass_rate.mean()),
    }


def _parse(spec: str) -> tuple[str, str, Path]:
    label, _, rest = spec.partition("=")
    kind, _, path = rest.partition(":")
    if not label or kind not in {"ppo", "bc"} or not path:
        raise ValueError(f"expected label=ppo:PATH or label=bc:PATH, got {spec!r}")
    return label, kind, Path(path)


def _load_all(specs: list[str]):
    parsed = [_parse(spec) for spec in specs]
    first_ppo = next((path for _, kind, path in parsed if kind == "ppo"), None)
    reference = first_ppo or parsed[0][2]
    reference_metadata = read_checkpoint_metadata(reference)
    model_config = ModelConfig(**reference_metadata["model_config"])
    model_config = replace(model_config, use_asymmetric_critic=True)
    model = create_model(model_config, reference_metadata.get("model_variant", "shared"))
    variables = P.initialize(model, jax.random.key(0))
    template = core.create_train_state(model, variables, core.PPOConfig())
    loaded = {}
    for label, kind, path in parsed:
        if kind == "bc":
            loaded[label] = _load_actor_checkpoint(path, variables, model_config)
        else:
            loaded[label] = _opponent_variables(path, template)
    return model, loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True, help="label=ppo|bc:PATH")
    parser.add_argument("--opponent", action="append", required=True, help="label=ppo|bc:PATH")
    parser.add_argument("--games-per-seat", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample", action="store_true", help="greedyではなく学習時と同じ温度でサンプリングする"
    )
    args = parser.parse_args()

    model, loaded = _load_all(args.candidate + args.opponent)
    seed_key = jax.random.key(args.seed)
    config = RolloutConfig()
    mode = "sampled" if args.sample else "greedy"
    print(f"games per pair: {2 * args.games_per_seat} (seed={args.seed}, {mode})")
    for candidate_spec in args.candidate:
        candidate_label = _parse(candidate_spec)[0]
        for opponent_spec in args.opponent:
            opponent_label = _parse(opponent_spec)[0]
            if candidate_label == opponent_label:
                continue
            result = play_paired(
                model,
                loaded[candidate_label],
                loaded[opponent_label],
                config,
                seed_key,
                args.games_per_seat,
                greedy=not args.sample,
            )
            print(
                f"{candidate_label} vs {opponent_label}: "
                f"score={result['score']:.3f} [{result['ci_low']:.3f}, {result['ci_high']:.3f}] "
                f"W/D/L={result['wins']}/{result['draws']}/{result['losses']} "
                f"seat0={result['seat0_score']:.2f} seat1={result['seat1_score']:.2f} "
                f"margin mean={result['margin_mean']:.0f} median={result['margin_median']:.0f} "
                f"pass={result['pass_rate']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
