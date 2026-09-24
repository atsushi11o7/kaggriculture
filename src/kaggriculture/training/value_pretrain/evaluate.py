"""保存済みcritic checkpoint(または未学習状態)を、再学習なしで評価するCLI。

例:
  uv run python -m kaggriculture.training.value_pretrain.evaluate \\
    outputs/value_pretrain/recent5_value/2026-09-20/00-25-26/checkpoints/best

  # 未学習(ランダム初期化)criticとの比較用
  uv run python -m kaggriculture.training.value_pretrain.evaluate --untrained \\
    --actor-checkpoint outputs/bc/recent5/2026-09-19/19-00-17/checkpoints/best
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import jax
import optax

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import policy as P
from kaggriculture.policy.jax.model_factory import create_model, critic_version
from kaggriculture.training.checkpoint import load_checkpoint, read_checkpoint_metadata
from kaggriculture.training.ppo.checkpointing import load_actor_checkpoint
from kaggriculture.training.value_pretrain import core
from kaggriculture.training.value_pretrain.evaluation import (
    compute_metrics,
    fit_linear_calibration,
    predict_values,
    sample_fixed_states,
    train_target_mean,
)


def _print_metrics(label: str, metrics: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  n_states={metrics['n_states']} n_values={metrics['n_values']}")
    print(f"  model_mse={metrics['model_mse']:.4f}")
    print(
        f"  zero_baseline_mse={metrics['zero_baseline_mse']:.4f}  "
        f"mean_baseline_mse={metrics['mean_baseline_mse']:.4f}"
    )
    print(
        f"  r2_vs_mean_baseline={metrics['r2_vs_mean_baseline']:.4f}  "
        f"correlation={metrics['correlation']:.4f}"
    )
    print(
        f"  sign_accuracy: all={metrics['sign_accuracy_all']:.3f} "
        f"|target|>=0.1={metrics['sign_accuracy_abs_target_ge_0.1']:.3f} "
        f"last5days={metrics['sign_accuracy_last5days']:.3f}"
    )
    print(
        f"  prediction: mean={metrics['prediction_mean']:.4f} std={metrics['prediction_std']:.4f}"
        f"   target: mean={metrics['target_mean']:.4f} std={metrics['target_std']:.4f}"
    )
    print("  day_mse:", {k: round(v, 4) for k, v in metrics["day_mse"].items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint", nargs="?", help="評価するvalue_pretrain checkpointのdirectory"
    )
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="checkpointの代わりに、指定actor+ランダム初期化criticを評価する",
    )
    parser.add_argument("--actor-checkpoint", type=str, default=None, help="--untrained時のActor")
    parser.add_argument(
        "--validation-dir", type=str, default="data/cache/value_pretrain/validation"
    )
    parser.add_argument("--train-dir", type=str, default="data/cache/value_pretrain/train")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--states-per-episode", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--turns-per-day", type=int, default=24)
    args = parser.parse_args()

    if args.untrained:
        if not args.actor_checkpoint:
            raise ValueError("--untrained requires --actor-checkpoint")
        actor_metadata = read_checkpoint_metadata(Path(args.actor_checkpoint))
        model_config = ModelConfig(**actor_metadata["model_config"])
        model = create_model(model_config)
        variables = P.initialize(model, jax.random.key(args.seed))
        variables = load_actor_checkpoint(Path(args.actor_checkpoint), variables, model_config)
        params = variables["params"]
        label = f"untrained critic (actor={args.actor_checkpoint})"
    else:
        if not args.checkpoint:
            raise ValueError("checkpoint path is required unless --untrained is given")
        ckpt = Path(args.checkpoint)
        metadata = json.loads((ckpt / "metadata.json").read_text())
        if metadata.get("critic_architecture_version") != critic_version():
            raise ValueError(f"incompatible critic architecture: {ckpt}")
        model_config = ModelConfig(**metadata["model_config"])
        model = create_model(model_config)
        variables = P.initialize(model, jax.random.key(args.seed))
        dummy_schedule = optax.cosine_decay_schedule(1.0, 1)
        state = core.create_train_state(
            model, variables, learning_rate=dummy_schedule, max_grad_norm=1.0
        )
        state, meta = load_checkpoint(ckpt, state)
        params = state.params
        label = f"{ckpt} (step={meta.get('step')})"

    validation_paths = [Path(p) for p in glob.glob(f"{args.validation_dir}/*.npz")]
    train_paths = [Path(p) for p in glob.glob(f"{args.train_dir}/*.npz")]
    print(f"validation shards available: {len(validation_paths)}")
    print(f"train shards available: {len(train_paths)}")

    mean = train_target_mean(train_paths)
    print(f"train target mean (constant baseline): {mean}")

    def fixed_set(paths):
        return sample_fixed_states(
            paths,
            episodes=args.episodes,
            states_per_episode=args.states_per_episode,
            seed=args.seed,
            turns_per_day=args.turns_per_day,
        )

    kwargs = {"train_mean": mean, "turns_per_day": args.turns_per_day}
    validation_set = fixed_set(validation_paths)
    train_set = fixed_set(train_paths)

    train_pred = predict_values(model, params, train_set.states, turns_per_day=args.turns_per_day)
    calibration = fit_linear_calibration(train_pred, train_set.targets)
    print(f"linear calibration fitted on train: a={calibration[0]:.4f} b={calibration[1]:.4f}")

    _print_metrics(
        f"{label} | VALIDATION raw", compute_metrics(model, params, validation_set, **kwargs)
    )
    _print_metrics(
        f"{label} | VALIDATION calibrated",
        compute_metrics(model, params, validation_set, calibration=calibration, **kwargs),
    )
    _print_metrics(f"{label} | TRAIN raw", compute_metrics(model, params, train_set, **kwargs))
    _print_metrics(
        f"{label} | TRAIN calibrated",
        compute_metrics(model, params, train_set, calibration=calibration, **kwargs),
    )


if __name__ == "__main__":
    main()
