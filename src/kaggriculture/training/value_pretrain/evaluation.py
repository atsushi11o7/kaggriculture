"""固定した試合・局面集合でcriticを評価する共通ロジック。

epochごとの学習時評価と、保存済みcheckpointを再学習なしで評価するCLIの両方から
呼び出される単一の実装。試合単位でサンプリングし、同一試合内の局面へ評価が
偏らないようにする。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

from kaggriculture.policy.jax import policy as P
from kaggriculture.simulator.state import State
from kaggriculture.training.value_pretrain.cache import load_shard

_DAY_BUCKETS = ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30))


@dataclass(frozen=True)
class FixedEvaluationSet:
    """固定サンプリング済みの評価用state/target/day。"""

    states: State
    targets: np.ndarray  # [N, 2]
    days: np.ndarray  # [N]、1始まり


def sample_fixed_states(
    shard_paths: list[Path],
    *,
    episodes: int | None,
    states_per_episode: int,
    seed: int,
    turns_per_day: int = 24,
) -> FixedEvaluationSet:
    """試合単位で固定抽出し、各試合からほぼ等間隔にstates_per_episode局面を取る。

    Args:
        shard_paths: 評価対象の試合shard(1試合=1ファイル)。
        episodes: 使用する試合数。Noneなら全試合。
        states_per_episode: 1試合あたりに抽出する局面数。
        seed: 試合選択に使う乱数シード(状態抽出自体は等間隔で決定論的)。
        turns_per_day: 1日あたりのターン数。

    Returns:
        FixedEvaluationSet: 全試合から集めたstate/target/dayの結合。
    """
    ordered = sorted(str(p) for p in shard_paths)
    if episodes is not None and episodes < len(ordered):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(ordered), size=episodes, replace=False)
        indices.sort()
        ordered = [ordered[i] for i in indices]

    all_states = []
    all_targets = []
    all_days = []
    for path in ordered:
        states, targets = load_shard(Path(path))
        length = len(targets)
        if length == 0:
            continue
        take = min(states_per_episode, length)
        pick = np.linspace(0, length - 1, take).astype(int)
        all_states.append(jax.tree.map(lambda x, pick=pick: x[pick], states))
        all_targets.append(targets[pick])
        all_days.append(np.asarray(states.step)[pick] // turns_per_day + 1)

    states = jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *all_states)
    targets = np.concatenate(all_targets, axis=0)
    days = np.concatenate(all_days, axis=0)
    return FixedEvaluationSet(states, targets, days)


def _safe_corrcoef(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.std() == 0 or target.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1])


def predict_values(
    model, params, states: State, *, turns_per_day: int = 24, batch_size: int = 512
) -> np.ndarray:
    """stateごとの価値予測[N, 2]を返す。"""
    n = len(states.step)
    preds = []
    for i in range(0, n, batch_size):
        batch = jax.tree.map(lambda x, i=i: x[i : i + batch_size], states)
        preds.append(
            np.asarray(
                P.state_values(model, {"params": params}, batch, turns_per_day=turns_per_day)
            )
        )
    return np.concatenate(preds, axis=0)


def fit_linear_calibration(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """target ~= a * pred + b の最小二乗解(a, b)を返す。"""
    a, b = np.polyfit(pred.reshape(-1), target.reshape(-1), 1)
    return float(a), float(b)


def compute_metrics(
    model,
    params,
    evaluation_set: FixedEvaluationSet,
    *,
    train_mean: np.ndarray,
    turns_per_day: int = 24,
    batch_size: int = 512,
    calibration: tuple[float, float] | None = None,
) -> dict:
    """固定評価集合に対するMSE・ベースライン・R²等を返す。

    calibrationを渡すと、予測をa*pred+bへ線形校正してから評価する。
    """
    targets, days = evaluation_set.targets, evaluation_set.days
    n = len(targets)
    pred = predict_values(
        model, params, evaluation_set.states, turns_per_day=turns_per_day, batch_size=batch_size
    )
    if calibration is not None:
        pred = calibration[0] * pred + calibration[1]

    flat_pred = pred.reshape(-1)
    flat_target = targets.reshape(-1)

    def mse(p, t):
        return float(np.mean((p - t) ** 2))

    zero_mse = mse(np.zeros_like(flat_target), flat_target)
    train_mean_broadcast = np.tile(train_mean, n)
    mean_mse = mse(train_mean_broadcast, flat_target)
    model_mse = mse(flat_pred, flat_target)
    r2_vs_mean_baseline = 1 - model_mse / max(mean_mse, 1e-12)
    correlation = _safe_corrcoef(flat_pred, flat_target)

    def sign_accuracy(p, t):
        if len(t) == 0:
            return float("nan")
        return float(np.mean(np.sign(p) == np.sign(t)))

    sign_all = sign_accuracy(flat_pred, flat_target)
    strong = np.abs(flat_target) >= 0.1
    sign_strong = sign_accuracy(flat_pred[strong], flat_target[strong])
    last5_mask = np.repeat(days >= 26, targets.shape[1])
    sign_last5 = sign_accuracy(flat_pred[last5_mask], flat_target[last5_mask])

    day_mse = {}
    for lo, hi in _DAY_BUCKETS:
        row_mask = (days >= lo) & (days <= hi)
        if not row_mask.any():
            day_mse[f"day{lo}-{hi}"] = float("nan")
            continue
        day_mse[f"day{lo}-{hi}"] = mse(pred[row_mask], targets[row_mask])

    return {
        "n_states": n,
        "n_values": len(flat_target),
        "model_mse": model_mse,
        "zero_baseline_mse": zero_mse,
        "mean_baseline_mse": mean_mse,
        "r2_vs_mean_baseline": r2_vs_mean_baseline,
        "correlation": correlation,
        "sign_accuracy_all": sign_all,
        "sign_accuracy_abs_target_ge_0.1": sign_strong,
        "sign_accuracy_last5days": sign_last5,
        "prediction_mean": float(flat_pred.mean()),
        "prediction_std": float(flat_pred.std()),
        "target_mean": float(flat_target.mean()),
        "target_std": float(flat_target.std()),
        "day_mse": day_mse,
    }


def train_target_mean(shard_paths: list[Path], sample: int = 300) -> np.ndarray:
    """train shardの一部からtargetの平均(定数ベースライン用)を推定する。"""
    ordered = sorted(str(p) for p in shard_paths)[:sample]
    totals = np.zeros(2, dtype=np.float64)
    count = 0
    for path in ordered:
        _, targets = load_shard(Path(path))
        totals += targets.sum(axis=0)
        count += len(targets)
    return (totals / max(count, 1)).astype(np.float32)
