"""Shared embedding and Encoder components for the CPU policy backend."""

import torch
import torch.nn as nn

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.torch import features as F


def _causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """Trueを未来位置に置くbool因果マスク。"""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


class TokenEmbedding(nn.Module):
    """語彙全体(policy.common.vocab)を共有する埋め込み。盤面トークンにも行動候補にも使う。"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.bag = nn.EmbeddingBag(V.VOCAB_SIZE, d_model, mode="sum")
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, index: torch.Tensor, value: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """collate済みの疎特徴量を埋め込む。

        Args:
            index: 語彙index。
            value: indexごとの重み。
            offset: 各SparseVectorの開始位置。

        Returns:
            形状(len(offset), d_model)の埋め込み。
        """
        return self.norm(self.bag(index, offset, value))

    def embed(self, vectors: list[V.SparseVector]) -> torch.Tensor:
        """SparseVector列をモデルと同じデバイスで埋め込む。

        Args:
            vectors: 埋め込む疎特徴量。

        Returns:
            形状(len(vectors), d_model)の埋め込み。
        """
        device = self.bag.weight.device
        index, value, offset = F.collate(vectors, device=device)
        return self(index, value, offset)


class Encoder(nn.Module):
    """盤面全体(疎特徴量のトークン列)を自己注意し、局面全体の集約表現を作る。"""

    def __init__(
        self,
        token_embedding: TokenEmbedding,
        board_position_embedding: nn.Embedding,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        owner_ids, zone_ids, position_ids = zip(*L.TOKEN_OWNER_ZONE_POSITION_WITH_CLS, strict=True)
        self.register_buffer("_owner_ids", torch.tensor(owner_ids, dtype=torch.long))
        self.register_buffer("_zone_ids", torch.tensor(zone_ids, dtype=torch.long))
        self.register_buffer("_position_ids", torch.tensor(position_ids, dtype=torch.long))
        self.owner_embedding = nn.Embedding(L.N_OWNERS, d_model)
        self.zone_embedding = nn.Embedding(L.N_ZONES, d_model)
        # Decoderと同じ盤面位置埋め込みを共有する。
        self.position_embedding = board_position_embedding
        # 正規化済みの内容表現を覆わないよう小さく初期化する。
        for embedding in (self.owner_embedding, self.zone_embedding):
            nn.init.normal_(embedding.weight, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, d_feedforward, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers, enable_nested_tensor=False
        )

    def forward(
        self, index: torch.Tensor, value: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """トークン列を自己注意し、[CLS]込みの系列全体の出力を返す。

        Args:
            index, value, offset: vocab.collate(全局面分のトークンを連結したもの)の出力。
                offsetの長さはbatch_size * L.NUM_WORDS_ENCODER。

        Returns:
            torch.Tensor: 形状(batch, L.NUM_WORDS_ENCODER + 1, d_model)。
            [:, 0]がCLSトークンの出力(局面全体の集約表現)。
        """
        embedded = self.token_embedding(index, value, offset)
        embedded = embedded.reshape(-1, L.NUM_WORDS_ENCODER, self.d_model)
        return self.forward_embedded(embedded)

    def forward_embedded(self, embedded: torch.Tensor) -> torch.Tensor:
        """Encode an already embedded fixed-width observation.

        Args:
            embedded: Tensor with shape ``(batch, tokens, d_model)``.

        Returns:
            Encoder output including the CLS token.
        """
        batch_size = embedded.size(0)
        cls = self.cls_token.expand(batch_size, 1, -1)
        hidden = torch.cat([cls, embedded], dim=1)
        hidden = hidden + self.owner_embedding(self._owner_ids)
        hidden = hidden + self.zone_embedding(self._zone_ids)
        hidden = hidden + self.position_embedding(self._position_ids)
        return self.transformer(hidden)


class PrivilegedEncoder(nn.Module):
    """両者の非公開状態を非対称critic用の小さな系列として符号化する。

    unit inventoryには公開位置を加え、誰がどこで何を持つかを保持する。
    """

    def __init__(
        self,
        token_embedding: TokenEmbedding,
        board_position_embedding: nn.Embedding,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.position_embedding = board_position_embedding
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        owner_ids, zone_ids = zip(*L.PRIVILEGED_OWNER_ZONE_WITH_CLS, strict=True)
        self.register_buffer("_owner_ids", torch.tensor(owner_ids, dtype=torch.long))
        self.register_buffer("_zone_ids", torch.tensor(zone_ids, dtype=torch.long))
        self.owner_embedding = nn.Embedding(L.N_OWNERS, d_model)
        self.zone_embedding = nn.Embedding(L.N_ZONES, d_model)
        for embedding in (self.owner_embedding, self.zone_embedding):
            nn.init.normal_(embedding.weight, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model, num_heads, d_feedforward, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

    def forward(
        self,
        index: torch.Tensor,
        value: torch.Tensor,
        offset: torch.Tensor,
        position_ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """非公開token列を符号化する。

        Args:
            index: 語彙index。
            value: indexごとの重み。
            offset: 各tokenの開始位置。
            position_ids: unitの盤面位置。
            key_padding_mask: 存在しないhandのmask。

        Returns:
            CLSを含む非公開Encoder出力。
        """
        batch_size = position_ids.size(0)
        x = self.token_embedding(index, value, offset)
        x = x.reshape(batch_size, L.NUM_PRIVILEGED_TOKENS, self.d_model)
        cls = self.cls_token.expand(batch_size, 1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_position = torch.full(
            (batch_size, 1), L.NO_POSITION, dtype=torch.long, device=position_ids.device
        )
        all_positions = torch.cat([cls_position, position_ids], dim=1)
        x = x + self.owner_embedding(self._owner_ids)
        x = x + self.zone_embedding(self._zone_ids)
        x = x + self.position_embedding(all_positions)

        cls_padding = torch.zeros((batch_size, 1), dtype=torch.bool, device=key_padding_mask.device)
        all_padding = torch.cat([cls_padding, key_padding_mask], dim=1)
        return self.transformer(x, src_key_padding_mask=all_padding)
