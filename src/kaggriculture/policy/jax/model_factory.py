"""Canonical V3 model construction."""

from kaggriculture.policy.common.config import CRITIC_ARCHITECTURE_VERSION, ModelConfig
from kaggriculture.policy.jax.model import PolicyValueNet


def create_model(config: ModelConfig) -> PolicyValueNet:
    """Create the only supported policy architecture."""
    return PolicyValueNet(config)


def critic_version() -> int:
    """Return the canonical critic architecture version."""
    return CRITIC_ARCHITECTURE_VERSION
