"""JAX PPO checkpointのactorを提出用PyTorch checkpointへ変換する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import torch

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import model as JM
from kaggriculture.policy.torch import model as TM
from kaggriculture.training.checkpoint import load_checkpoint
from kaggriculture.training.ppo.core import PPOConfig, create_train_state
from kaggriculture.training.weight_bridge import jax_to_torch


def export_policy(source: Path, destination: Path) -> None:
    """非対称criticを除外し、提出に必要なactor重みだけを書き出す。"""
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    model_config = ModelConfig(**metadata["model_config"])
    ppo_config = PPOConfig(**metadata["ppo_config"])
    model = JM.PolicyValueNet(model_config)
    variables = JM.initialize(model, jax.random.key(0))
    target = create_train_state(model, variables, ppo_config)
    restored, _ = load_checkpoint(source, target)

    actor_config = {
        "d_model": model_config.d_model,
        "num_heads": model_config.num_heads,
        "d_feedforward": model_config.d_feedforward,
        "num_layers_encoder": model_config.num_layers_encoder,
        "num_layers_decoder": model_config.num_layers_decoder,
        "dropout": model_config.dropout,
        "use_episode_history": model_config.use_episode_history,
        "use_asymmetric_critic": False,
        "num_layers_critic": model_config.num_layers_critic,
    }
    actor = TM.PolicyValueNet(ModelConfig(**actor_config))
    jax_to_torch({"params": restored.params}, actor, actor_only=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": actor.state_dict(),
            "config": {"model": actor_config},
            "source_update": metadata["update"],
        },
        destination,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="JAX checkpoint directory")
    parser.add_argument("destination", type=Path, help="output .pt path")
    args = parser.parse_args()
    export_policy(args.source, args.destination)


if __name__ == "__main__":
    main()
