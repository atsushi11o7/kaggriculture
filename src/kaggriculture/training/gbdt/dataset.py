"""リプレイから合法候補単位のランキングデータを構築する。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kaggriculture.policy.torch.candidate_api import (
    CandidateDecision,
    trace_expert_candidates,
)
from kaggriculture.training.gbdt.features import FEATURE_NAMES, context_features, group_features
from kaggriculture.training.replays.io import iter_replay_samples


@dataclass
class RankingData:
    """LightGBMへ渡す候補行、正解ラベル、query group。"""

    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray


@dataclass
class BuildStats:
    """GBDT候補データ構築の成功数と破棄理由。"""

    accepted: int = 0
    discarded: int = 0
    groups: int = 0
    reasons: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "discarded": self.discarded,
            "groups": self.groups,
            "reasons": dict(sorted(self.reasons.items())),
        }

    @classmethod
    def from_dict(cls, value: dict) -> BuildStats:
        return cls(
            value.get("accepted", 0),
            value.get("discarded", 0),
            value.get("groups", 0),
            Counter(value.get("reasons", {})),
        )


def _discard_reason(error: Exception) -> str:
    if isinstance(error, KeyError):
        return "missing_field"
    if isinstance(error, TypeError):
        return "malformed_type"
    if "quantity" in str(error):
        return "invalid_quantity"
    if "not found among" in str(error):
        return "illegal_candidate"
    return "invalid_value"


def _groups_for_sample(obs: dict, action: dict, rules: dict, total_days: int):
    groups = []

    def visit(decision: CandidateDecision, selected: int) -> None:
        context = context_features(
            obs,
            decision.context,
            decision.kind,
            decision.position,
            decision.board_position,
            total_days,
        )
        matrix = group_features(
            context, decision.candidates, decision.context, decision.board_position
        )
        groups.append((decision.kind, matrix, selected))

    trace_expert_candidates(obs, action, visit, **rules)
    return groups


def build_ranking_data(
    episode_files: list[Path | tuple[Path, tuple[int, ...]]],
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1.0,
    max_market_orders: int = 10,
    min_player_reward: float | None = None,
    episode_steps: int = 720,
    stats: BuildStats | None = None,
) -> dict[str, RankingData]:
    """複数リプレイをスロット種別ごとのランキングデータへ変換する。"""
    rows = {kind: [] for kind in ("unit_op", "market_op", "quantity")}
    labels = {kind: [] for kind in rows}
    groups = {kind: [] for kind in rows}
    rules = {
        "turns_per_day": turns_per_day,
        "shed_capacity": shed_capacity,
        "hire_mult": hire_mult,
        "max_market_orders": max_market_orders,
    }
    total_days = (episode_steps + turns_per_day - 1) // turns_per_day
    stats = stats if stats is not None else BuildStats()
    for item in episode_files:
        path, selected_players = item if isinstance(item, tuple) else (item, None)
        for obs, action in iter_replay_samples(
            path, min_player_reward, set(selected_players) if selected_players else None
        ):
            try:
                sample_groups = _groups_for_sample(obs, action, rules, total_days)
                for kind, matrix, selected in sample_groups:
                    target = np.zeros(len(matrix), dtype=np.int8)
                    target[selected] = 1
                    rows[kind].append(matrix)
                    labels[kind].append(target)
                    groups[kind].append(len(matrix))
                stats.accepted += 1
                stats.groups += len(sample_groups)
            except (KeyError, TypeError, ValueError) as error:
                stats.discarded += 1
                stats.reasons[_discard_reason(error)] += 1
    result = {}
    for kind in rows:
        if not rows[kind]:
            continue
        result[kind] = RankingData(
            np.concatenate(rows[kind]),
            np.concatenate(labels[kind]),
            np.asarray(groups[kind], dtype=np.int32),
        )
    if not result:
        raise ValueError("no representable ranking groups were found")
    return result


def save_ranking_data(
    path: Path, data: dict[str, RankingData], stats: BuildStats | None = None
) -> None:
    """ランキングデータを再利用可能な圧縮NPZへ保存する。"""
    arrays = {}
    for kind, value in data.items():
        arrays[f"{kind}__features"] = value.features
        arrays[f"{kind}__labels"] = value.labels
        arrays[f"{kind}__groups"] = value.groups
    arrays["metadata"] = np.asarray(
        json.dumps(
            {"feature_names": FEATURE_NAMES, "stats": stats.to_dict() if stats else {}},
            sort_keys=True,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_ranking_data(path: Path) -> dict[str, RankingData]:
    """保存済みランキングデータを読み込む。"""
    result = {}
    with np.load(path, allow_pickle=False) as arrays:
        for kind in ("unit_op", "market_op", "quantity"):
            key = f"{kind}__features"
            if key in arrays:
                result[kind] = RankingData(
                    arrays[key], arrays[f"{kind}__labels"], arrays[f"{kind}__groups"]
                )
        metadata = json.loads(str(arrays["metadata"]))
        if tuple(metadata["feature_names"]) != FEATURE_NAMES:
            raise ValueError("ranking cache feature contract does not match this code")
    return result


def load_ranking_stats(path: Path) -> BuildStats:
    """ranking cacheに保存された構築統計を返す。"""
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"]))
    return BuildStats.from_dict(metadata.get("stats", {}))
