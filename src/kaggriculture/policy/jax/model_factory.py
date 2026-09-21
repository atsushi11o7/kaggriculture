"""共有版と完全分離critic版の明示的なモデル生成。"""

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax.model import PolicyValueNet
from kaggriculture.policy.jax.separated_model import SeparatedPolicyValueNet

SHARED = "shared"
SEPARATED = "separated"


def create_model(config: ModelConfig, variant: str = SHARED):
    """設定されたcritic実装のPolicyValueNetを生成する。

    Args:
        config: 共通のActor/Critic形状設定。
        variant: ``shared``または``separated``。

    Returns:
        Flaxモデル。

    Raises:
        ValueError: 未知のvariant、または分離版で非対称criticが無効な場合。
    """
    if variant == SHARED:
        return PolicyValueNet(config)
    if variant == SEPARATED:
        if not config.use_asymmetric_critic:
            raise ValueError("separated model requires use_asymmetric_critic=true")
        return SeparatedPolicyValueNet(config)
    raise ValueError(f"unknown model variant: {variant}")


def critic_version(variant: str) -> int:
    """Return the checkpoint version for a critic implementation."""
    if variant == SHARED:
        from kaggriculture.policy.common.config import CRITIC_ARCHITECTURE_VERSION

        return CRITIC_ARCHITECTURE_VERSION
    if variant == SEPARATED:
        from kaggriculture.policy.jax.separated_model import (
            SEPARATED_CRITIC_ARCHITECTURE_VERSION,
        )

        return SEPARATED_CRITIC_ARCHITECTURE_VERSION
    raise ValueError(f"unknown model variant: {variant}")


def metadata_variant(metadata: dict) -> str:
    """Read a checkpoint variant while treating legacy checkpoints as shared."""
    return metadata.get("model_variant", SHARED)
