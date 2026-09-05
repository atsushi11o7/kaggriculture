"""方策(policy)・価値(value)・相手shed推定を出力するネットワークの本体。

エンコーダは盤面全体(自分の100マス+相手の100マス+その他のゾーン)を疎特徴量
(policy.vocab.SparseVector)としてnn.EmbeddingBagで埋め込み、TransformerEncoderで
自己注意する。各トークンの owner(誰の情報か)・zone(何の種類の情報か)・
position(タイルの場合の盤面上の位置)は局面によらず固定なので、埋め込みバッグの
外側で別途加算する。

デコーダはfarmer→hand1→hand2→…→市場注文1→…の順に決定スロットを並べ、
「それまでに確定した決定」を1つ右にずらして入力する標準的なTransformerデコーダ
(因果マスク付き自己注意+エンコーダ出力への交差注意)。各スロットの出力隠れ状態を
使い、そのスロットで今合法な候補(actions.py参照)だけをスコアリングする
(score_candidates)。存在しない候補にはそもそもスコアという概念が無いため、
「合法候補だけから選ぶ」という構造は保証される。

ただしこのクラス自体が保証するのは合法性だけで、複数スロットの同時決定としての
整合性(farmerとhandが同じタイルを取り合う、同じ品目のshed在庫を奪い合う、等)
までは保証しない。それはactions.pyへ渡す状態をdecode_state.pyで逐次更新する
ことで担保する(呼び出し側の責務)。

エンコーダとデコーダは同じ語彙埋め込み(TokenEmbedding)を共有する。「MELONを
植える」という行動候補と「タイルにMELONが植わっている」という盤面事実が同じ
埋め込みを使うことで、観測と行動の対応関係を学習しやすくする狙い
(vocab.pyのACTION_FARMER_OP等のコメント参照)。
"""

import torch
import torch.nn as nn

from kaggriculture.policy import vocab as V
from kaggriculture.policy.model_config import (
    D_FEEDFORWARD,
    D_MODEL,
    DROPOUT,
    NUM_HEADS,
    NUM_LAYERS_DECODER,
    NUM_LAYERS_ENCODER,
)
from kaggriculture.simulator import constants as C

BOARD_SIZE = 10
N_TILE_TOKENS = 2 * BOARD_SIZE * BOARD_SIZE  # 自分の盤面 + 相手の盤面

# 1ターンの決定スロット数の上限: farmer(1) + hand(最大MAX_HANDS) +
# 市場注文(最大MAX_MARKET_ORDERS件 + 「もう注文しない」の1候補)。
MAX_DECODE_LEN = 1 + C.MAX_HANDS + (C.MAX_MARKET_ORDERS + 1)

# --- エンコーダ側トークンのowner/zone/position ---
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


def _causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """位置iがi以下しか参照できない加算マスク(未来位置は-inf)。"""
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)


class TokenEmbedding(nn.Module):
    """語彙全体(policy.vocab)を共有する埋め込み。盤面トークンにも行動候補にも使う。"""

    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.bag = nn.EmbeddingBag(V.VOCAB_SIZE, d_model, mode="sum")
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, index: torch.Tensor, value: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """SparseVectorをvocab.collateで平坦化した(index, value, offset)を埋め込む。

        Returns:
            torch.Tensor: 形状(len(offset), d_model)。
        """
        return self.norm(self.bag(index, offset, value))


class Encoder(nn.Module):
    """盤面全体(疎特徴量のトークン列)を自己注意し、局面全体の集約表現を作る。"""

    def __init__(
        self,
        token_embedding: TokenEmbedding,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        d_feedforward: int = D_FEEDFORWARD,
        num_layers: int = NUM_LAYERS_ENCODER,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.d_model = d_model
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
            index, value, offset: vocab.collate(全局面分のトークンを連結したもの)の出力。
                offsetの長さはbatch_size * NUM_WORDS_ENCODER。

        Returns:
            torch.Tensor: 形状(batch, NUM_WORDS_ENCODER + 1, d_model)。
            [:, 0]がCLSトークンの出力(局面全体の集約表現)。
        """
        x = self.token_embedding(index, value, offset)
        x = x.reshape(-1, NUM_WORDS_ENCODER, self.d_model)
        batch_size = x.size(0)

        cls = self.cls_token.expand(batch_size, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.owner_embedding(self._owner_ids)
        x = x + self.zone_embedding(self._zone_ids)
        x = x + self.position_embedding(self._position_ids)

        return self.transformer(x)


class Decoder(nn.Module):
    """farmer→hands→市場注文の順に、決定スロットの隠れ状態を作る。

    標準的なTransformerデコーダのteacher forcing形式: それまでに確定した決定
    (教師強制時は正解、推論時はサンプリング済みの候補)を1つ右にずらして入力し、
    因果マスク付き自己注意+エンコーダ出力への交差注意で、次のスロットを判断する
    ための隠れ状態を出す。候補の集合はスロットごとに大きさが異なるため、
    このクラス自体は候補を知らない(スコアリングはscore_candidatesで別途行う)。
    """

    def __init__(
        self,
        token_embedding: TokenEmbedding,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        d_feedforward: int = D_FEEDFORWARD,
        num_layers: int = NUM_LAYERS_DECODER,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.d_model = d_model
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # 位置0が<START>(farmerスロットの判断に使う)、位置iがスロットi-1確定後
        # (=スロットiの判断に使う)の隠れ状態に対応する。
        self.position_embedding = nn.Embedding(MAX_DECODE_LEN + 1, d_model)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        self.policy_proj = nn.Linear(d_model, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, d_feedforward, dropout, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(
        self,
        memory: torch.Tensor,
        decision_index: torch.Tensor,
        decision_value: torch.Tensor,
        decision_offset: torch.Tensor,
        num_slots: int,
        decision_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """確定済み決定列(教師強制)からスロットごとの判断用隠れ状態を計算する。

        Args:
            memory: 形状(batch, S_enc, d_model)。Encoder.forwardの出力。
            decision_index, decision_value, decision_offset: vocab.collateの出力
                (バッチ全体でnum_slots個ずつの決定トークンを連結したもの)。
                パディング位置は空のSparseVector(index=[], value=[])にする。
            num_slots: このバッチの決定スロット数(<START>を含まない、バッチ内最大)。
            decision_padding_mask: 形状(batch, num_slots)、Trueがパディング位置。
                Noneならパディング無し(全example同じスロット数)。

        Returns:
            torch.Tensor: 形状(batch, num_slots + 1, d_model)。位置iの出力が
            「スロットi(0-indexed、0=farmer)の候補をスコアリングするための隠れ状態」。
        """
        batch_size = memory.size(0)
        dec_tokens = self.token_embedding(decision_index, decision_value, decision_offset)
        dec_tokens = dec_tokens.reshape(batch_size, num_slots, self.d_model)

        start = self.start_token.expand(batch_size, 1, -1)
        tgt = torch.cat([start, dec_tokens], dim=1)
        position_ids = torch.arange(num_slots + 1, device=tgt.device)
        tgt = tgt + self.position_embedding(position_ids)

        tgt_key_padding_mask = None
        if decision_padding_mask is not None:
            start_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=tgt.device)
            tgt_key_padding_mask = torch.cat([start_pad, decision_padding_mask], dim=1)

        out = self.transformer(
            tgt,
            memory,
            tgt_mask=_causal_mask(num_slots + 1, device=tgt.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return out

    def score_candidates(
        self, hidden: torch.Tensor, candidates: list[V.SparseVector]
    ) -> torch.Tensor:
        """1スロット分の隠れ状態から、今合法な候補それぞれのスコアを計算する。

        Args:
            hidden: 形状(d_model,)。forwardの出力のうち、このスロットに対応する1点。
            candidates: このスロットで今合法な候補(actions.py参照)。候補数は
                スロットごとに異なるため、バッチ化はしない(呼び出し側でスロット
                ごとにこの関数を呼ぶ)。

        Returns:
            torch.Tensor: 形状(len(candidates),)。候補ごとの生スコア(logit)。
            呼び出し側でsoftmax/cross_entropyする。
        """
        index, value, offset = V.collate(candidates)
        index, value, offset = (
            index.to(hidden.device),
            value.to(hidden.device),
            offset.to(hidden.device),
        )
        cand_emb = self.token_embedding(index, value, offset)  # (len(candidates), d_model)
        return cand_emb @ self.policy_proj(hidden)


class PolicyValueNet(nn.Module):
    """方策・価値・相手shed推定・市場注文個数を出力するネットワーク。"""

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        d_feedforward: int = D_FEEDFORWARD,
        num_layers_encoder: int = NUM_LAYERS_ENCODER,
        num_layers_decoder: int = NUM_LAYERS_DECODER,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(d_model)
        self.encoder = Encoder(
            self.token_embedding, d_model, num_heads, d_feedforward, num_layers_encoder, dropout
        )
        self.decoder = Decoder(
            self.token_embedding, d_model, num_heads, d_feedforward, num_layers_decoder, dropout
        )

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
        # 市場注文の個数(n)を予測するヘッド。選んだ(op, item)候補が決まってから、
        # そのスロットの隠れ状態+選んだ候補の埋め込みを材料に、正規化した個数
        # ([0, 1]、呼び出し側で「買える/売れる最大数」等にスケールし直す想定)を
        # 回帰する。farmer/handの候補には個数の概念が無いため、このヘッドは
        # item_indexを持つ市場注文候補(BUY_SEED/BUY_ANIMAL/BUY_PRODUCT/SELL)
        # にのみ使う。
        self.quantity_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def encode(
        self, index: torch.Tensor, value: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """盤面トークン列からエンコーダ出力([CLS]込み)を計算する。"""
        return self.encoder(index, value, offset)

    def value_and_shed(self, encoder_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """エンコーダ出力から価値と相手shed推定を計算する。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (value, shed_logits)。
                value: 形状(batch, 1)。
                shed_logits: 形状(batch, N_SHED_ITEMS, n_shed_buckets)の生ロジット
                    (呼び出し側でsoftmax/cross_entropyする前提)。
        """
        cls_out = encoder_out[:, 0]
        value_out = self.value_head(cls_out)
        shed_logits = self.shed_head(cls_out).reshape(-1, len(V.SHED_ITEM), self.n_shed_buckets)
        return value_out, shed_logits

    def predict_quantity(
        self, hidden: torch.Tensor, chosen_candidate: V.SparseVector
    ) -> torch.Tensor:
        """選んだ市場注文候補の個数(正規化済み、[0, 1]目安)を予測する。

        Args:
            hidden: 形状(d_model,)。そのスロットのDecoder隠れ状態。
            chosen_candidate: そのスロットで実際に選んだ候補(item_indexを持つもの)。

        Returns:
            torch.Tensor: 形状(1,)。sigmoidをかける前の生の値(呼び出し側でsigmoid)。
        """
        index, value, offset = V.collate([chosen_candidate])
        index, value, offset = (
            index.to(hidden.device),
            value.to(hidden.device),
            offset.to(hidden.device),
        )
        cand_emb = self.token_embedding(index, value, offset)[0]  # (d_model,)
        return self.quantity_head(torch.cat([hidden, cand_emb], dim=-1))
