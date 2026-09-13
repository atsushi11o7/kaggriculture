"""Torch/JAXで共有するモデル構成の型。値はHydraまたはcheckpointから与える。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """重み形状とActor/Critic入力を決めるモデル構成。"""

    d_model: int
    num_heads: int
    d_feedforward: int
    num_layers_encoder: int
    num_layers_decoder: int
    dropout: float
    use_episode_history: bool
    use_asymmetric_critic: bool
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
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
