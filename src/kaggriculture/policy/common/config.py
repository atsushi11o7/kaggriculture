"""Torch/JAXで共有するモデル構成の型。値はHydraまたはcheckpointから与える。"""

from dataclasses import dataclass

from kaggriculture.rules import constants as C

POLICY_ARCHITECTURE = "fixed_slot_parallel_v3"
POLICY_ARCHITECTURE_VERSION = 3
CRITIC_ARCHITECTURE_VERSION = 3
NUM_CRITIC_MACRO_FEATURES = 12
CRITIC_PARAMETER_MODULES = frozenset(
    {
        "value_head",
        "privileged_encoder",
        "critic_macro_encoder",
    }
)


def checkpoint_shape_metadata() -> dict[str, int | str]:
    """checkpoint互換性を決める固定shape情報を返す。"""
    return {
        "architecture": POLICY_ARCHITECTURE,
        "architecture_version": POLICY_ARCHITECTURE_VERSION,
        "max_hands": C.MAX_HANDS,
    }


def validate_checkpoint_metadata(metadata: dict) -> None:
    """異なる方策schemaや固定shapeのcheckpointを早期に拒否する。"""
    expected = checkpoint_shape_metadata()
    actual = {name: metadata.get(name) for name in expected}
    if actual != expected:
        raise ValueError(f"incompatible policy checkpoint: expected {expected}, got {actual}")


@dataclass(frozen=True)
class ModelConfig:
    """重み形状とActor/Critic入力を決めるモデル構成。"""

    d_model: int
    num_heads: int
    d_feedforward: int
    num_layers_encoder: int
    num_layers_decoder: int
    num_layers_critic: int

    def __post_init__(self) -> None:
        """Transformerの形状と設定値を構築時に検証する。"""
        if self.d_model <= 0 or self.num_heads <= 0 or self.d_model % self.num_heads:
            raise ValueError("positive d_model must be divisible by num_heads")
        for name in (
            "d_feedforward",
            "num_layers_encoder",
            "num_layers_decoder",
            "num_layers_critic",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
