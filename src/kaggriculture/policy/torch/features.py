"""PyTorch向けの疎特徴量batch変換。"""

import torch

from kaggriculture.policy.common.vocab import SparseVector


def collate(
    vectors: list[SparseVector], device: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """疎特徴量列をEmbeddingBag用の平坦なtensorへ変換する。

    Args:
        vectors: 変換する疎特徴量。
        device: 出力tensorの配置先。

    Returns:
        語彙index、特徴量の重み、各vectorの開始offset。
    """
    index: list[int] = []
    value: list[float] = []
    offset: list[int] = []
    cursor = 0
    for vector in vectors:
        offset.append(cursor)
        index.extend(vector.index)
        value.extend(vector.value)
        cursor += len(vector.index)
    return (
        torch.tensor(index, dtype=torch.long, device=device),
        torch.tensor(value, dtype=torch.float32, device=device),
        torch.tensor(offset, dtype=torch.long, device=device),
    )
