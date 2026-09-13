"""SparseVector列をJAX/PyTorch共通の固定形状へ詰める。"""

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from kaggriculture.policy.common import vocab as V

if TYPE_CHECKING:
    from numpy.typing import NDArray

MAX_ENCODER_FEATURES = 32
MAX_PRIVILEGED_FEATURES = 24
MAX_DECODER_FEATURES = 32
MAX_CANDIDATE_FEATURES = 2


class DenseFeatures(NamedTuple):
    """固定幅にpaddingした疎特徴量。末尾のmaskはvalue=0で表す。"""

    index: "NDArray[np.int32]"
    value: "NDArray[np.float32]"


def pack_vectors(vectors: list[V.SparseVector], width: int) -> DenseFeatures:
    """SparseVector列を`(n_vectors, width)`へ変換する。

    Args:
        vectors: 変換する疎特徴量。
        width: 1トークン当たりの固定特徴数。

    Returns:
        固定形状のindexとvalue。

    Raises:
        ValueError: いずれかのトークンがwidthを超える場合。
    """
    index = np.zeros((len(vectors), width), dtype=np.int32)
    value = np.zeros((len(vectors), width), dtype=np.float32)
    for row, vector in enumerate(vectors):
        n = len(vector.index)
        if n > width:
            raise ValueError(f"sparse vector has {n} features, width={width}")
        index[row, :n] = vector.index
        value[row, :n] = vector.value
    return DenseFeatures(index=index, value=value)


def pack_batch(batch: list[list[V.SparseVector]], width: int) -> DenseFeatures:
    """同じ系列長のSparseVector列をバッチ化する。"""
    if not batch:
        raise ValueError("batch must not be empty")
    seq_len = len(batch[0])
    if any(len(sequence) != seq_len for sequence in batch):
        raise ValueError("all sequences must have the same length")
    packed = [pack_vectors(sequence, width) for sequence in batch]
    return DenseFeatures(
        index=np.stack([features.index for features in packed]),
        value=np.stack([features.value for features in packed]),
    )


def pack_padded_batch(
    batch: list[list[V.SparseVector]], width: int
) -> tuple[DenseFeatures, "NDArray[np.bool_]"]:
    """可変長系列を最大系列長までpaddingしてバッチ化する。"""
    if not batch:
        raise ValueError("batch must not be empty")
    max_len = max(len(sequence) for sequence in batch)
    index = np.zeros((len(batch), max_len, width), dtype=np.int32)
    value = np.zeros((len(batch), max_len, width), dtype=np.float32)
    padding = np.ones((len(batch), max_len), dtype=np.bool_)
    for row, sequence in enumerate(batch):
        packed = pack_vectors(sequence, width)
        n = len(sequence)
        index[row, :n] = packed.index
        value[row, :n] = packed.value
        padding[row, :n] = False
    return DenseFeatures(index=index, value=value), padding
