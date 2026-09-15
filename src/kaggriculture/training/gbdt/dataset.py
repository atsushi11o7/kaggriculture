"""リプレイから合法候補単位のランキングデータを構築する。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kaggriculture.policy.torch.candidate_api import (
    CandidateDecision,
    candidate_action_names,
    trace_expert_candidates,
)
from kaggriculture.training.gbdt.features import FEATURE_NAMES, context_features, group_features
from kaggriculture.training.replays.io import iter_replay_samples

# market_opはSTOP/HIREだけで全決定の過半数を占め、LambdaRankの比較勾配がそこへ
# 支配されてBUY_SEED等の少数派だが決定的に重要な行動をほぼ選べなくなる(実際に
# 自己対戦で種を一度も購入せず資金が尽きる問題を確認した)。そのためquery単位の
# クラス重みで補正する。unit_opの低精度クラス(移動方向)は多数派崩壊ではなく
# 位置依存の難しさが原因なので、対象外にする。quantityは離散クラスを持たない。
_WEIGHTED_KIND = "market_op"


@dataclass
class RankingData:
    """LightGBMへ渡す候補行、正解ラベル、query group。"""

    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    # query(決定)ごとの重み。同じqueryに属する全行へ同じ値を入れる。多数派崩壊を
    # 補正しないkind(unit_op/quantity)ではNone(=一律1.0扱い)。
    weights: np.ndarray | None = None


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


def _quantity_indices(count: int, selected: int, limit: int) -> np.ndarray:
    """正例・境界・近傍・対数間隔を保つ決定論的な数量負例選択。"""
    if count <= limit:
        return np.arange(count, dtype=np.int32)
    if limit < 3:
        raise ValueError("quantity candidate limit must be at least 3")
    keep = {0, count - 1, selected}
    for delta in (1, -1, 2, -2):
        if len(keep) >= limit:
            break
        index = selected + delta
        if 0 <= index < count:
            keep.add(index)
    for index in np.geomspace(1, count, num=max(limit * 2, 8)):
        if len(keep) >= limit:
            break
        keep.add(min(count - 1, int(round(index)) - 1))
    for index in np.linspace(0, count - 1, num=count, dtype=int):
        if len(keep) >= limit:
            break
        keep.add(int(index))
    return np.asarray(sorted(keep), dtype=np.int32)


def _op_class(kind: str, candidate) -> str | None:
    """重み付け対象kindのみ、選ばれた候補のop名(重みのクラスキー)を返す。"""
    if kind != _WEIGHTED_KIND:
        return None
    op, _item = candidate_action_names(candidate, kind)
    return op


def _balanced_class_weights(codes: list[int], num_classes: int) -> np.ndarray:
    """出現頻度の平方根に反比例するquery重みを、平均1.0になるよう正規化して返す。

    完全な逆頻度(sklearnの'balanced'相当)だと出現数が数十件しかないクラス
    (例: BUY_LAND)の重みが数十倍になり、少数例への過学習を招く。平方根で
    緩和し、[0.25, 8.0]でさらにクリップして安定させる。
    """
    counts = np.bincount(codes, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    raw = total / (num_classes * np.maximum(counts, 1))
    weight = np.clip(np.sqrt(raw), 0.25, 8.0)
    mean_weight = float(np.sum(weight * counts) / total)
    return (weight / mean_weight).astype(np.float32)


def _groups_for_sample(
    obs: dict, action: dict, rules: dict, total_days: int, quantity_limit: int | None = None
):
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
        candidates = decision.candidates
        weight_class = _op_class(decision.kind, candidates[selected])
        if decision.kind == "quantity" and quantity_limit is not None:
            indices = _quantity_indices(len(candidates), selected, quantity_limit)
            candidates = [candidates[int(index)] for index in indices]
            selected = int(np.searchsorted(indices, selected))
        matrix = group_features(context, candidates, decision.context, decision.board_position)
        groups.append((decision.kind, matrix, selected, weight_class))

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
    quantity_limit: int | None = None,
) -> dict[str, RankingData]:
    """複数リプレイをスロット種別ごとのランキングデータへ変換する。"""
    rows = {kind: [] for kind in ("unit_op", "market_op", "quantity")}
    labels = {kind: [] for kind in rows}
    groups = {kind: [] for kind in rows}
    weight_codes: dict[str, list[int]] = {kind: [] for kind in rows}
    class_vocabs: dict[str, dict[str, int]] = {kind: {} for kind in rows}
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
                sample_groups = _groups_for_sample(obs, action, rules, total_days, quantity_limit)
                for kind, matrix, selected, weight_class in sample_groups:
                    target = np.zeros(len(matrix), dtype=np.int8)
                    target[selected] = 1
                    rows[kind].append(matrix)
                    labels[kind].append(target)
                    groups[kind].append(len(matrix))
                    if weight_class is not None:
                        vocab = class_vocabs[kind]
                        weight_codes[kind].append(vocab.setdefault(weight_class, len(vocab)))
                stats.accepted += 1
                stats.groups += len(sample_groups)
            except (KeyError, TypeError, ValueError) as error:
                stats.discarded += 1
                stats.reasons[_discard_reason(error)] += 1
    result = {}
    for kind in rows:
        if not rows[kind]:
            continue
        group_arr = np.asarray(groups[kind], dtype=np.int32)
        weights = None
        if weight_codes[kind]:
            per_class = _balanced_class_weights(weight_codes[kind], len(class_vocabs[kind]))
            weights = np.concatenate(
                [
                    np.full(int(size), per_class[code], dtype=np.float32)
                    for size, code in zip(group_arr, weight_codes[kind], strict=True)
                ]
            )
        result[kind] = RankingData(
            np.concatenate(rows[kind]),
            np.concatenate(labels[kind]),
            group_arr,
            weights,
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
        if value.weights is not None:
            arrays[f"{kind}__weights"] = value.weights
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
                weight_key = f"{kind}__weights"
                result[kind] = RankingData(
                    arrays[key],
                    arrays[f"{kind}__labels"],
                    arrays[f"{kind}__groups"],
                    arrays[weight_key] if weight_key in arrays else None,
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


@dataclass(frozen=True)
class RankingFiles:
    """種別ごとの生配列。読み込みはmemmapで行う。"""

    directory: Path
    kinds: dict[str, dict[str, int]]
    stats: BuildStats


def _write_group(streams: dict, matrix: np.ndarray, selected: int) -> None:
    matrix.astype(np.float32, copy=False).tofile(streams["features"])
    label = np.zeros(len(matrix), dtype=np.int8)
    label[selected] = 1
    label.tofile(streams["labels"])
    np.asarray([len(matrix)], dtype=np.int32).tofile(streams["groups"])


def build_ranking_files(
    episode_files: list[Path | tuple[Path, tuple[int, ...]]],
    directory: Path,
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1.0,
    max_market_orders: int = 10,
    min_player_reward: float | None = None,
    episode_steps: int = 720,
    quantity_limit: int = 24,
) -> RankingFiles:
    """候補行をRAMに蓄積せず、種別別の連続バイナリへ逐次保存する。"""
    if quantity_limit < 3:
        raise ValueError("quantity_limit must be at least 3")
    directory.mkdir(parents=True, exist_ok=True)
    kinds = {kind: {"rows": 0, "groups": 0} for kind in ("unit_op", "market_op", "quantity")}
    stats = BuildStats()
    # 行の特徴量本体はRAMへ蓄積しないが、query単位の重み計算に要るクラスコード
    # (kindあたり8種類程度の小さな整数)だけは全件保持しても軽い。
    weight_codes: dict[str, list[int]] = {kind: [] for kind in kinds}
    class_vocabs: dict[str, dict[str, int]] = {kind: {} for kind in kinds}
    rules = {
        "turns_per_day": turns_per_day,
        "shed_capacity": shed_capacity,
        "hire_mult": hire_mult,
        "max_market_orders": max_market_orders,
    }
    total_days = (episode_steps + turns_per_day - 1) // turns_per_day
    from contextlib import ExitStack

    with ExitStack() as stack:
        streams = {
            kind: {
                name: stack.enter_context(open(directory / f"{kind}.{name}.bin", "wb"))
                for name in ("features", "labels", "groups")
            }
            for kind in kinds
        }
        for item in episode_files:
            path, selected_players = item if isinstance(item, tuple) else (item, None)
            for obs, action in iter_replay_samples(
                path, min_player_reward, set(selected_players) if selected_players else None
            ):
                try:
                    sample_groups = _groups_for_sample(
                        obs, action, rules, total_days, quantity_limit
                    )
                    for kind, matrix, selected, weight_class in sample_groups:
                        _write_group(streams[kind], matrix, selected)
                        kinds[kind]["rows"] += len(matrix)
                        kinds[kind]["groups"] += 1
                        if weight_class is not None:
                            vocab = class_vocabs[kind]
                            weight_codes[kind].append(vocab.setdefault(weight_class, len(vocab)))
                    stats.accepted += 1
                    stats.groups += len(sample_groups)
                except (KeyError, TypeError, ValueError) as error:
                    stats.discarded += 1
                    stats.reasons[_discard_reason(error)] += 1
    if not any(value["groups"] for value in kinds.values()):
        raise ValueError("no representable ranking groups were found")
    for kind, codes in weight_codes.items():
        if not codes:
            continue
        # 全件を見てから初めてクラス頻度が確定するため、features/labels/groupsを
        # 書き終えた後にweights.binだけ別途書き出す(1queryにつき1回のクラス
        # コードなので、この時点でも行数分の重みをRAMへ展開するのはこのkindの
        # rows分だけで済む)。
        per_class = _balanced_class_weights(codes, len(class_vocabs[kind]))
        group_sizes = np.fromfile(directory / f"{kind}.groups.bin", dtype=np.int32)
        with open(directory / f"{kind}.weights.bin", "wb") as f:
            for code, size in zip(codes, group_sizes, strict=True):
                np.full(int(size), per_class[code], dtype=np.float32).tofile(f)
    metadata = {
        "feature_names": FEATURE_NAMES,
        "quantity_limit": quantity_limit,
        "kinds": kinds,
        "stats": stats.to_dict(),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, sort_keys=True))
    return RankingFiles(directory, kinds, stats)


def load_ranking_files(directory: Path) -> RankingFiles:
    """逐次書き出した候補行をメモリマップ用に開く。"""
    metadata = json.loads((directory / "metadata.json").read_text())
    if tuple(metadata["feature_names"]) != FEATURE_NAMES:
        raise ValueError("ranking cache feature contract does not match this code")
    return RankingFiles(directory, metadata["kinds"], BuildStats.from_dict(metadata["stats"]))


def open_ranking_data(files: RankingFiles, kind: str) -> RankingData:
    """評価用に指定種別だけをmemmapで参照する。"""
    info = files.kinds[kind]
    rows, groups = info["rows"], info["groups"]
    if not rows:
        raise ValueError(f"no {kind} groups")
    directory = files.directory
    weights_path = directory / f"{kind}.weights.bin"
    weights = (
        np.memmap(weights_path, dtype=np.float32, mode="r", shape=(rows,))
        if weights_path.exists()
        else None
    )
    return RankingData(
        np.memmap(
            directory / f"{kind}.features.bin",
            dtype=np.float32,
            mode="r",
            shape=(rows, len(FEATURE_NAMES)),
        ),
        np.memmap(directory / f"{kind}.labels.bin", dtype=np.int8, mode="r", shape=(rows,)),
        np.memmap(directory / f"{kind}.groups.bin", dtype=np.int32, mode="r", shape=(groups,)),
        weights,
    )
