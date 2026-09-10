"""Behavior Cloningの目的関数。"""

from collections.abc import Sequence

import torch


def mean_token_nll(log_probs: torch.Tensor, num_decisions: Sequence[int]) -> torch.Tensor:
    """複合行動のlog probabilityからtoken平均NLLを計算する。

    Args:
        log_probs: サンプルごとに全決定を合計したlog probability。
        num_decisions: サンプルごとの決定数。

    Returns:
        batch内の全決定に対する平均negative log likelihood。

    Raises:
        ValueError: 入力形状が不正、または決定数が正でない場合。
    """
    if log_probs.ndim != 1:
        raise ValueError("log_probs must have shape (batch,)")
    if len(num_decisions) != log_probs.shape[0]:
        raise ValueError("num_decisions length must match log_probs")
    total_decisions = sum(num_decisions)
    if total_decisions <= 0 or any(n <= 0 for n in num_decisions):
        raise ValueError("num_decisions must contain only positive values")
    return -log_probs.sum() / total_decisions
