"""PPO系toolのcheckpoint互換性検証とparameter復元。"""

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
    """checkpointが現在のcriticアーキテクチャを使っているか検証する。"""
    if metadata.get("critic_architecture_version") != critic_version():
        raise ValueError(f"incompatible critic architecture: {path}")


def load_actor_checkpoint(path: Path, variables: dict, target_config: ModelConfig) -> dict:
    """対象criticを維持したまま、互換性のあるActor parameterを復元する。

    Args:
        path: checkpointディレクトリ。
        variables: critic parameterを含む初期化済みの対象variables。
        target_config: 対象モデル構成。

    Returns:
        復元したActorと初期化済み対象criticを含むvariables。

    Raises:
        ValueError: Actorの構成、名前、shapeに互換性がない場合。
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
    """互換性のある共同学習済みActor・critic checkpointを復元する。

    Args:
        path: value pretrainingまたは共同BCのcheckpointディレクトリ。
        variables: 復元先モデルの初期化済みvariables。
        target_config: 復元先のモデル構成。
        gamma: PPOで使用する割引率。
        daily_reward_coefficient: PPOの日次報酬係数。
        daily_reward_scale: PPOの日次報酬正規化尺度。
        daily_reward_maximum: PPOの日次報酬上限。
        allow_reward_mismatch: Trueなら教師報酬設定の不一致を許可する。

    Returns:
        checkpointから全parameterを復元したvariables。

    Raises:
        ValueError: trainer、critic、報酬、モデル構成、parameterに互換性がない場合。
    """
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


def load_policy_value_checkpoint(path: Path, variables: dict, target_config: ModelConfig) -> dict:
    """互換checkpointからActor・criticの全parameterを復元する。

    Args:
        path: 復元元のcheckpointディレクトリ。
        variables: 復元先モデルの初期化済みvariables。
        target_config: 復元先のモデル構成。

    Returns:
        checkpointから全parameterを復元したvariables。

    Raises:
        ValueError: critic version、モデル構成、parameter名・shapeに互換性がない場合。
    """
    metadata = read_checkpoint_metadata(path)
    validate_checkpoint_metadata(metadata)
    validate_critic_architecture(metadata, path)
    if ModelConfig(**metadata["model_config"]) != target_config:
        raise ValueError(f"incompatible policy/value checkpoint config: {path}")
    saved = serialization.msgpack_restore((path / "state.msgpack").read_bytes())
    source = traverse_util.flatten_dict(saved["params"])
    target = traverse_util.flatten_dict(unfreeze(variables["params"]))
    if source.keys() != target.keys() or any(
        source[name].shape != target[name].shape for name in source.keys() & target.keys()
    ):
        raise ValueError(f"incompatible policy/value checkpoint parameters: {path}")
    restored = {name: jnp.asarray(source[name]) for name in target}
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
    """BC checkpointをPPOの初期Actor/Criticとして読み込む。

    Args:
        path: BC checkpointディレクトリ。
        variables: PPOモデルの初期化済みvariables。
        target_config: PPOモデル構成。
        gamma: PPOで使用する割引率。
        daily_reward_coefficient: PPOの日次報酬係数。
        daily_reward_scale: PPOの日次報酬正規化尺度。
        daily_reward_maximum: PPOの日次報酬上限。
        allow_reward_mismatch: Trueなら共同学習criticの報酬設定不一致を許可する。

    Returns:
        共同value学習済みならActor・critic、そうでなければActorだけを復元したvariables。
    """
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
    """固定PPO対戦相手のモデルparameterだけを復元する。"""
    metadata = read_checkpoint_metadata(directory)
    return load_policy_value_checkpoint(
        directory,
        {"params": train_state.params},
        ModelConfig(**metadata["model_config"]),
    )
