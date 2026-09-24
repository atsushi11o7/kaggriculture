"""Checkpoint compatibility and parameter restoration for PPO-family tools."""

from pathlib import Path

import jax.numpy as jnp
from flax import serialization, traverse_util
from flax.core import freeze, unfreeze

from kaggriculture.policy.common.config import (
    CRITIC_PARAMETER_MODULES,
    ModelConfig,
    validate_checkpoint_metadata,
)
from kaggriculture.policy.jax.model_factory import critic_version
from kaggriculture.training.checkpoint import (
    load_checkpoint,
    read_checkpoint_metadata,
)

_ACTOR_CONFIG_FIELDS = (
    "d_model",
    "num_heads",
    "d_feedforward",
    "num_layers_encoder",
    "num_layers_decoder",
)


def validate_critic_architecture(metadata: dict, path: Path) -> None:
    """Validate that a checkpoint uses the current critic architecture."""
    if metadata.get("critic_architecture_version") != critic_version():
        raise ValueError(f"incompatible critic architecture: {path}")


def load_actor_checkpoint(path: Path, variables: dict, target_config: ModelConfig) -> dict:
    """Restore compatible Actor parameters while retaining the target critic.

    Args:
        path: Checkpoint directory.
        variables: Initialized target variables, including critic parameters.
        target_config: Target model configuration.

    Returns:
        Variables containing the restored Actor and initialized target critic.

    Raises:
        ValueError: If Actor configuration, names, or shapes are incompatible.
    """
    metadata = read_checkpoint_metadata(path)
    validate_checkpoint_metadata(metadata)
    source_config = ModelConfig(**metadata["model_config"])
    if any(
        getattr(source_config, field) != getattr(target_config, field)
        for field in _ACTOR_CONFIG_FIELDS
    ):
        raise ValueError(f"incompatible actor checkpoint config: {path}")
    saved = serialization.msgpack_restore((path / "state.msgpack").read_bytes())
    source = traverse_util.flatten_dict(saved["params"])
    target = traverse_util.flatten_dict(unfreeze(variables["params"]))

    def is_actor(name):
        return name[0] not in CRITIC_PARAMETER_MODULES

    source_actor = {name for name in source if is_actor(name)}
    target_actor = {name for name in target if is_actor(name)}
    if source_actor != target_actor or any(
        source[name].shape != target[name].shape for name in source_actor & target_actor
    ):
        raise ValueError(f"incompatible actor checkpoint: {path}")
    for name in target_actor:
        target[name] = jnp.asarray(source[name])
    return {"params": freeze(traverse_util.unflatten_dict(target))}


def load_value_checkpoint(
    path: Path,
    variables: dict,
    target_config: ModelConfig,
    gamma: float,
    daily_reward_coefficient: float,
    daily_reward_scale: float,
    daily_reward_maximum: float,
    *,
    allow_reward_mismatch: bool = False,
) -> dict:
    """Restore a compatible jointly trained Actor and critic checkpoint."""
    metadata = read_checkpoint_metadata(path)
    validate_checkpoint_metadata(metadata)
    is_value_checkpoint = metadata.get("trainer") == "value_pretrain" or (
        metadata.get("trainer") == "bc" and metadata.get("joint_value_training") is True
    )
    if not is_value_checkpoint:
        raise ValueError(f"not a value-pretraining checkpoint: {path}")
    validate_critic_architecture(metadata, path)
    reward_mode = metadata.get("reward_mode")
    if reward_mode == "terminal_win":
        source_reward = (0.0, 10000.0, 0.02)
    elif reward_mode == "terminal_win_daily_asset":
        source_reward = (
            float(metadata["daily_reward_coefficient"]),
            float(metadata["daily_reward_scale"]),
            float(metadata["daily_reward_maximum"]),
        )
    else:
        raise ValueError(f"unsupported value checkpoint reward: {path}")
    target_reward = daily_reward_coefficient, daily_reward_scale, daily_reward_maximum
    if abs(float(metadata["gamma"]) - gamma) > 1e-9:
        raise ValueError("value checkpoint gamma differs from PPO")
    reward_mismatch = any(
        abs(source - target) > 1e-9
        for source, target in zip(source_reward, target_reward, strict=True)
    )
    if reward_mismatch and not allow_reward_mismatch:
        raise ValueError("value checkpoint reward configuration differs from PPO")
    if ModelConfig(**metadata["model_config"]) != target_config:
        raise ValueError(f"incompatible value checkpoint model config: {path}")

    saved = serialization.msgpack_restore((path / "state.msgpack").read_bytes())
    source = traverse_util.flatten_dict(saved["params"])
    target = traverse_util.flatten_dict(unfreeze(variables["params"]))
    restored = {}
    for name, target_value in target.items():
        if name not in source or source[name].shape != target_value.shape:
            raise ValueError(f"incompatible value checkpoint parameters: {path}")
        restored[name] = jnp.asarray(source[name])
    return {"params": freeze(traverse_util.unflatten_dict(restored))}


def load_initial_bc_checkpoint(
    path: Path,
    variables: dict,
    target_config: ModelConfig,
    gamma: float,
    daily_reward_coefficient: float,
    daily_reward_scale: float,
    daily_reward_maximum: float,
    *,
    allow_reward_mismatch: bool = False,
) -> dict:
    """Load a BC Actor and preserve its jointly trained critic when available."""
    metadata = read_checkpoint_metadata(path)
    if metadata.get("trainer") == "bc" and metadata.get("joint_value_training") is True:
        return load_value_checkpoint(
            path,
            variables,
            target_config,
            gamma,
            daily_reward_coefficient,
            daily_reward_scale,
            daily_reward_maximum,
            allow_reward_mismatch=allow_reward_mismatch,
        )
    return load_actor_checkpoint(path, variables, target_config)


def load_opponent_variables(directory: Path, train_state) -> dict:
    """Restore checkpoint parameters using the given PPO train-state template."""
    metadata = read_checkpoint_metadata(directory)
    validate_checkpoint_metadata(metadata)
    validate_critic_architecture(metadata, directory)
    restored, _ = load_checkpoint(directory, train_state)
    return {"params": restored.params}
