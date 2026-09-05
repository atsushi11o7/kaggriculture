"""方策(policy)・価値(value)・相手shed推定を出力するネットワークの本体。

エンコーダは盤面全体(自分の100マス+相手の100マス+その他のゾーン)を疎特徴量
(policy.vocab.SparseVector)としてnn.EmbeddingBagで埋め込み、TransformerEncoderで
自己注意する。各トークンの owner(誰の情報か)・zone(何の種類の情報か)・
position(タイルの場合の盤面上の位置)は局面によらず固定なので、埋め込みバッグの
外側で別途加算する。

このモジュールはエンコーダ+価値ヘッド+相手shed推定ヘッドまで。方策(合法な
行動候補をスコアリングするデコーダ側)は別途追加する。
"""

import torch
import torch.nn as nn

from kaggriculture.policy import vocab as V
from kaggriculture.policy.model_config import (
    D_FEEDFORWARD,
    D_MODEL,
    DROPOUT,
    NUM_HEADS,
    NUM_LAYERS_ENCODER,
)

BOARD_SIZE = 10
N_TILE_TOKENS = 2 * BOARD_SIZE * BOARD_SIZE  # 自分の盤面 + 相手の盤面

# tokenize.get_encoder_inputが返すトークン列の並び順(CLSを先頭に足す前)に対応する
# (owner, zone)。ここがズレるとエンコーダが誤った文脈を学習するため、
# tokenize.pyの実装と一致させることが必須(test_model.pyで長さを検証する)。
_OWNER_SHARED, _OWNER_OWN, _OWNER_OPP = 0, 1, 2
N_OWNERS = 3

(
    _ZONE_CLS,
    _ZONE_TILE,
    _ZONE_PLAYER_INFO,
    _ZONE_SHED,
    _ZONE_SEEDS,
    _ZONE_INVENTORY,
    _ZONE_MARKET,
    _ZONE_TOWN,
    _ZONE_TURN,
) = range(9)
N_ZONES = 9

# tileトークンのみ意味を持つ位置(盤面上のy*board_size+x)。tile以外のトークンは
# この専用の埋め込み次元を使わないという意味で、有効なタイル位置の数(=1個余分な
# 添字)を「該当なし」に割り当てる。
_NO_POSITION = N_TILE_TOKENS // 2
N_POSITIONS = _NO_POSITION + 1

# tokenize.get_encoder_inputの並び順そのまま: 自分の盤面100 + 相手の盤面100 +
# player_info(自分・相手)+ shed(自分・相手)+ seeds + inventory + market + town + turn
_TOKEN_OWNER_ZONE_POSITION = (
    [(_OWNER_OWN, _ZONE_TILE, i) for i in range(_NO_POSITION)]
    + [(_OWNER_OPP, _ZONE_TILE, i) for i in range(_NO_POSITION)]
    + [
        (_OWNER_OWN, _ZONE_PLAYER_INFO, _NO_POSITION),
        (_OWNER_OPP, _ZONE_PLAYER_INFO, _NO_POSITION),
        (_OWNER_OWN, _ZONE_SHED, _NO_POSITION),
        (_OWNER_OPP, _ZONE_SHED, _NO_POSITION),
        (_OWNER_OWN, _ZONE_SEEDS, _NO_POSITION),
        (_OWNER_OWN, _ZONE_INVENTORY, _NO_POSITION),
        (_OWNER_SHARED, _ZONE_MARKET, _NO_POSITION),
        (_OWNER_SHARED, _ZONE_TOWN, _NO_POSITION),
        (_OWNER_SHARED, _ZONE_TURN, _NO_POSITION),
    ]
)
NUM_WORDS_ENCODER = len(_TOKEN_OWNER_ZONE_POSITION)  # tokenize.get_encoder_inputの出力トークン数

# CLSトークン(先頭に追加)の分を足した(owner, zone, position)の並び
_TOKEN_OWNER_ZONE_POSITION_WITH_CLS = [
    (_OWNER_SHARED, _ZONE_CLS, _NO_POSITION)
] + _TOKEN_OWNER_ZONE_POSITION


class Encoder(nn.Module):
    """盤面全体(疎特徴量のトークン列)を自己注意し、局面全体の集約表現を作る。"""

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        d_feedforward: int = D_FEEDFORWARD,
        num_layers: int = NUM_LAYERS_ENCODER,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.bag = nn.EmbeddingBag(V.VOCAB_SIZE, d_model, mode="sum")
        self.bag_norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        owner_ids, zone_ids, position_ids = zip(*_TOKEN_OWNER_ZONE_POSITION_WITH_CLS, strict=True)
        self.register_buffer("_owner_ids", torch.tensor(owner_ids, dtype=torch.long))
        self.register_buffer("_zone_ids", torch.tensor(zone_ids, dtype=torch.long))
        self.register_buffer("_position_ids", torch.tensor(position_ids, dtype=torch.long))
        self.owner_embedding = nn.Embedding(N_OWNERS, d_model)
        self.zone_embedding = nn.Embedding(N_ZONES, d_model)
        self.position_embedding = nn.Embedding(N_POSITIONS, d_model)
        # owner/zone/positionは正規化済みトークンへの加算なので、内容を覆い隠さない
        # 程度に小さく初期化する(参考にしたPolicyValueNetと同じ考え方)。
        for embedding in (self.owner_embedding, self.zone_embedding, self.position_embedding):
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
            index: バッチ全体のトークン疎特徴量を連結したindex列(SparseBatch参照)。
            value: 同上のvalue列。
            offset: 同上のoffset列(batch_size * NUM_WORDS_ENCODER個のトークン境界)。

        Returns:
            torch.Tensor: 形状(batch, NUM_WORDS_ENCODER + 1, d_model)。
            [:, 0]がCLSトークンの出力(局面全体の集約表現)。
        """
        x = self.bag_norm(self.bag(index, offset, value))
        x = x.reshape(-1, NUM_WORDS_ENCODER, self.d_model)
        batch_size = x.size(0)

        cls = self.cls_token.expand(batch_size, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.owner_embedding(self._owner_ids)
        x = x + self.zone_embedding(self._zone_ids)
        x = x + self.position_embedding(self._position_ids)

        return self.transformer(x)


class PolicyValueNet(nn.Module):
    """方策・価値・相手shed推定を出力するネットワーク(方策ヘッドは未実装)。"""

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        d_feedforward: int = D_FEEDFORWARD,
        num_layers_encoder: int = NUM_LAYERS_ENCODER,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(d_model, num_heads, d_feedforward, num_layers_encoder, dropout)

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1)
        )
        # 相手の納屋(SHED_ITEM語彙と同じN_SHED_ITEMS品目)ごとの分布。何個かの離散
        # バケットに分けたカテゴリ分布として出す(点推定ではなくサンプリング可能に
        # するため。詳細はbucketの設計を別途詰める)。
        n_shed_items = len(V.SHED_ITEM)
        self.n_shed_buckets = 16
        self.shed_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_shed_items * self.n_shed_buckets),
        )

    def forward(
        self, index: torch.Tensor, value: torch.Tensor, offset: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """バッチ分の盤面(疎特徴量)から、価値と相手shed推定を計算する。

        Args:
            index, value, offset: Encoder.forward参照。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (value, shed_logits)。
                value: 形状(batch, 1)。
                shed_logits: 形状(batch, N_SHED_ITEMS, n_shed_buckets)の生ロジット
                    (呼び出し側でsoftmax/cross_entropyする前提)。
        """
        encoder_out = self.encoder(index, value, offset)
        cls_out = encoder_out[:, 0]

        value_out = self.value_head(cls_out)
        shed_logits = self.shed_head(cls_out).reshape(-1, len(V.SHED_ITEM), self.n_shed_buckets)

        return value_out, shed_logits
