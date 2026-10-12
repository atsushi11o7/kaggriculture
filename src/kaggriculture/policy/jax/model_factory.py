"""V3モデルの正規構築経路。"""

from kaggriculture.policy.common.config import CRITIC_ARCHITECTURE_VERSION, ModelConfig
from kaggriculture.policy.jax.model import PolicyValueNet


def create_model(config: ModelConfig) -> PolicyValueNet:
    """唯一サポートする方策アーキテクチャを構築する。"""
    return PolicyValueNet(config)


def critic_version() -> int:
    """正規criticアーキテクチャのversionを返す。"""
    return CRITIC_ARCHITECTURE_VERSION
