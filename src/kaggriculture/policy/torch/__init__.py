"""提出用CPU backendとリプレイ候補処理。"""

from .model import PolicyValueNet
from .policy import predict_action

__all__ = ["PolicyValueNet", "predict_action"]
