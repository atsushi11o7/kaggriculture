"""GBDT候補rankerで既存の自己回帰デコードを駆動する。"""

from __future__ import annotations

import numpy as np

from kaggriculture.policy.torch.candidate_api import CandidateDecision, decode_with_candidates
from kaggriculture.training.gbdt.features import context_features, group_features
from kaggriculture.training.gbdt.model import GBDTRanker


class GBDTAgent:
    """Transformerを使わず、GBDTスコアだけで1ターンを生成する教師。"""

    def __init__(
        self,
        ranker: GBDTRanker,
        *,
        turns_per_day: int = 24,
        shed_capacity: int = 100,
        hire_mult: float = 1.0,
        max_market_orders: int = 10,
        episode_steps: int = 720,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.ranker = ranker
        self.rules = {
            "turns_per_day": turns_per_day,
            "shed_capacity": shed_capacity,
            "hire_mult": hire_mult,
            "max_market_orders": max_market_orders,
        }
        self.total_days = (episode_steps + turns_per_day - 1) // turns_per_day
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def _choose(self, scores: np.ndarray) -> int:
        if self.temperature <= 0:
            return int(np.argmax(scores))
        logits = (scores - scores.max()) / self.temperature
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(len(scores), p=probabilities))

    def act(self, observation: dict) -> dict:
        """観測からKaggle形式の複合行動を生成する。"""

        def choose(decision: CandidateDecision) -> int:
            context = context_features(
                observation,
                decision.context,
                decision.kind,
                decision.position,
                decision.board_position,
                self.total_days,
            )
            features = group_features(
                context, decision.candidates, decision.context, decision.board_position
            )
            return self._choose(self.ranker.score(decision.kind, features))

        return decode_with_candidates(observation, choose, **self.rules)

    def __call__(self, observation, configuration=None) -> dict:
        """kaggle-environmentsのagent callable契約。"""
        return self.act(observation)
