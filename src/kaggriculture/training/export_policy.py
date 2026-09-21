"""Export a parallel JAX actor as a CPU-only PyTorch checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import jax
import optax
import torch

from kaggriculture.policy.common.config import (
    ModelConfig,
    checkpoint_shape_metadata,
    validate_checkpoint_metadata,
)
from kaggriculture.policy.jax.model_factory import create_model
from kaggriculture.policy.jax.policy import initialize
from kaggriculture.policy.torch.model import PolicyValueNet as TorchNet
from kaggriculture.training.bc import core as bc_core
from kaggriculture.training.checkpoint import load_checkpoint
from kaggriculture.training.ppo import core as ppo_core
from kaggriculture.training.weight_bridge import jax_to_torch


def export_policy(source: Path, destination: Path) -> None:
    """Strip the training-only critic and save the parallel actor for submission."""
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    validate_checkpoint_metadata(metadata)
    training_config = ModelConfig(**metadata["model_config"])
    training_model = create_model(training_config, metadata.get("model_variant", "shared"))
    variables = initialize(training_model, jax.random.key(0))
    config = metadata["config"]
    rules = config["rules"]
    if metadata.get("trainer") == "bc":
        train = config["train"]
        # bc/train.pyはlearning_rateを常にcosine decay schedule(callable)として
        # optimizerへ渡すため、optax内部のstate構造(step数を追跡するcount等)は
        # 定数floatの場合と異なる。値そのものはこの後すぐparamsだけ取り出して
        # 捨てるので、復元用テンプレートは構造さえ一致すればよい(ダミーの
        # schedule)。
        optimizer = bc_core.BCConfig(
            learning_rate=optax.cosine_decay_schedule(train["learning_rate"], 1),
            weight_decay=train["weight_decay"],
            max_grad_norm=train["max_grad_norm"],
            turns_per_day=rules["turns_per_day"],
            shed_capacity=rules["shed_capacity"],
        )
        target = bc_core.create_train_state(training_model, variables, optimizer)
        source_step = metadata["step"]
    elif metadata.get("trainer") == "ppo":
        ppo = config["ppo"]
        fields = ppo_core.PPOConfig.__dataclass_fields__
        optimizer = ppo_core.PPOConfig(
            **{name: ppo[name] for name in fields if name in ppo},
            turns_per_day=rules["turns_per_day"],
            shed_capacity=rules["shed_capacity"],
        )
        target = ppo_core.create_train_state(training_model, variables, optimizer)
        source_step = metadata["update"]
    else:
        raise ValueError(f"unsupported trainer: {metadata.get('trainer')!r}")
    restored, _ = load_checkpoint(source, target)

    actor_config = replace(training_config, use_asymmetric_critic=False)
    actor = TorchNet(actor_config)
    jax_to_torch({"params": restored.params}, actor, actor_only=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint_shape_metadata(),
            "model_state_dict": actor.state_dict(),
            "config": {"model": asdict(actor_config)},
            "source_step": source_step,
        },
        destination,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export_policy(args.source, args.destination)


if __name__ == "__main__":
    main()
