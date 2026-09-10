"""共有埋め込みとTransformerによる自己回帰型の方策・価値ネットワーク。

公開盤面をEncoderで符号化し、Decoderがfarmer、hands、市場注文を順に生成する。
各スロットでは合法候補だけを採点し、選択後の仮状態更新はdistribution.pyが担う。
品目と盤面位置の埋め込みは観測・行動間で共有する。
"""

import torch
import torch.nn as nn

from kaggriculture.policy import token_layout as L
from kaggriculture.policy import vocab as V
from kaggriculture.policy.model_config import (
    D_FEEDFORWARD,
    D_MODEL,
    DROPOUT,
    NUM_HEADS,
    NUM_LAYERS_CRITIC,
    NUM_LAYERS_DECODER,
    NUM_LAYERS_ENCODER,
)


def _causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """Trueを未来位置に置くbool因果マスク。"""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


class TokenEmbedding(nn.Module):
    """語彙全体(policy.vocab)を共有する埋め込み。盤面トークンにも行動候補にも使う。"""

    def __init__(self, d_model: int = D_MODEL) -> None:
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
        index, value, offset = V.collate(vectors, device=device)
        return self(index, value, offset)


class Encoder(nn.Module):
    """盤面全体(疎特徴量のトークン列)を自己注意し、局面全体の集約表現を作る。"""

    def __init__(
        self,
        token_embedding: TokenEmbedding,
        board_position_embedding: nn.Embedding,
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
        x = self.token_embedding(index, value, offset)
        x = x.reshape(-1, L.NUM_WORDS_ENCODER, self.d_model)
        batch_size = x.size(0)

        cls = self.cls_token.expand(batch_size, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.owner_embedding(self._owner_ids)
        x = x + self.zone_embedding(self._zone_ids)
        x = x + self.position_embedding(self._position_ids)

        return self.transformer(x)


class Decoder(nn.Module):
    """farmer→hands→市場注文を因果的に符号化するTransformer Decoder。

    入力は直前の決定と現在スロットのユニット情報。可変な合法候補は別途採点する。
    """

    def __init__(
        self,
        token_embedding: TokenEmbedding,
        board_position_embedding: nn.Embedding,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        d_feedforward: int = D_FEEDFORWARD,
        num_layers: int = NUM_LAYERS_DECODER,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.d_model = d_model
        # 決定順ではなく、スロット種別ごとに固定した位置ID。
        self.position_embedding = nn.Embedding(L.MAX_DECODE_LEN, d_model)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        # Encoderのタイルと同じ盤面位置埋め込みを共有する。
        self.board_position_embedding = board_position_embedding
        self.policy_proj = nn.Linear(d_model, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, d_feedforward, dropout, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(
        self,
        memory: torch.Tensor,
        tgt_index: torch.Tensor,
        tgt_value: torch.Tensor,
        tgt_offset: torch.Tensor,
        position_ids: torch.Tensor,
        board_position_ids: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """決定トークン列から各スロットの隠れ状態を計算する。

        Args:
            memory: 形状(batch, encoder_len, d_model)のEncoder出力。
            tgt_index: 決定トークンの語彙index。
            tgt_value: indexごとの重み。
            tgt_offset: 各決定トークンの開始位置。
            position_ids: 形状(batch, seq_len)の固定スロット位置。
            board_position_ids: 形状(batch, seq_len)のunit盤面位置。
            tgt_key_padding_mask: Trueがpaddingのmask。

        Returns:
            形状(batch, seq_len, d_model)の隠れ状態。
        """
        batch_size, seq_len = position_ids.shape
        tgt = self.token_embedding(tgt_index, tgt_value, tgt_offset)
        tgt = tgt.reshape(batch_size, seq_len, self.d_model)
        tgt = tgt + self.position_embedding(position_ids.to(tgt.device))
        tgt = tgt + self.board_position_embedding(board_position_ids.to(tgt.device))

        return self.transformer(
            tgt,
            memory,
            tgt_mask=_causal_mask(seq_len, device=tgt.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

    def score_candidates(
        self, hidden: torch.Tensor, candidates: list[V.SparseVector]
    ) -> torch.Tensor:
        """1スロットの合法候補を採点する。

        Args:
            hidden: 形状(d_model,)のスロット表現。
            candidates: 合法候補。

        Returns:
            候補ごとのlogit。
        """
        cand_emb = self.token_embedding.embed(candidates)  # (len(candidates), d_model)
        # LayerNorm後の内積が過大にならないようattentionと同じ尺度にする。
        return (cand_emb @ self.policy_proj(hidden)) / (self.d_model**0.5)

    def score_candidates_batch(
        self, hidden: torch.Tensor, candidates_per_row: list[list[V.SparseVector]]
    ) -> torch.Tensor:
        """複数スロットの可変数候補をまとめて採点する。

        Args:
            hidden: 形状(batch, d_model)のスロット表現。
            candidates_per_row: 行ごとの合法候補。

        Returns:
            padding列が-infの候補logit。
        """
        n = hidden.shape[0]
        max_cands = max(len(c) for c in candidates_per_row)
        flat_cands: list[V.SparseVector] = []
        valid_counts = []
        for cands in candidates_per_row:
            flat_cands.extend(cands)
            flat_cands.extend(V.SparseVector() for _ in range(max_cands - len(cands)))
            valid_counts.append(len(cands))

        cand_emb = self.token_embedding.embed(flat_cands).view(n, max_cands, self.d_model)
        proj = self.policy_proj(hidden).unsqueeze(-1)  # (n, d_model, 1)
        scores = torch.bmm(cand_emb, proj).squeeze(-1) / (self.d_model**0.5)  # (n, max_cands)
        col_idx = torch.arange(max_cands, device=hidden.device).unsqueeze(0)
        counts = torch.tensor(valid_counts, device=hidden.device).unsqueeze(1)
        return scores.masked_fill(col_idx >= counts, float("-inf"))


class PrivilegedEncoder(nn.Module):
    """両者の非公開状態を非対称critic用の小さな系列として符号化する。

    unit inventoryには公開位置を加え、誰がどこで何を持つかを保持する。
    """

    def __init__(
        self,
        token_embedding: TokenEmbedding,
        board_position_embedding: nn.Embedding,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        d_feedforward: int = D_FEEDFORWARD,
        num_layers: int = NUM_LAYERS_CRITIC,
        dropout: float = DROPOUT,
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


class PolicyValueNet(nn.Module):
    """方策・価値を出力するネットワーク。数量もscore_candidatesでスコアリングする
    候補の1種として方策に含まれる(専用ヘッドは持たない)。"""

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        d_feedforward: int = D_FEEDFORWARD,
        num_layers_encoder: int = NUM_LAYERS_ENCODER,
        num_layers_decoder: int = NUM_LAYERS_DECODER,
        dropout: float = DROPOUT,
        use_episode_history: bool = False,
        use_asymmetric_critic: bool = False,
        num_layers_critic: int = NUM_LAYERS_CRITIC,
    ) -> None:
        """ネットワークを構築する。

        Args:
            d_model: 埋め込み次元。
            num_heads: attention head数。
            d_feedforward: feed-forward層の中間次元。
            num_layers_encoder: 公開Encoderの層数。
            num_layers_decoder: Decoderの層数。
            dropout: Transformerのdropout率。
            use_episode_history: 累積実績をactor入力に含める。Trueなら各APIで
                countersが必須。
            use_asymmetric_critic: critic専用の非公開情報Encoderを追加する。actorの
                推論には使われないが、共有埋め込みと公開Encoderにはvalue lossも流れる。
            num_layers_critic: 非公開情報Encoderの層数。
        """
        super().__init__()
        self.uses_episode_history = use_episode_history
        self.uses_asymmetric_critic = use_asymmetric_critic
        self.token_embedding = TokenEmbedding(d_model)
        # EncoderのタイルとDecoderのunit位置で共有する。
        self.board_position_embedding = nn.Embedding(L.N_POSITIONS, d_model)
        nn.init.normal_(self.board_position_embedding.weight, std=0.02)
        self.encoder = Encoder(
            self.token_embedding,
            self.board_position_embedding,
            d_model,
            num_heads,
            d_feedforward,
            num_layers_encoder,
            dropout,
        )
        self.decoder = Decoder(
            self.token_embedding,
            self.board_position_embedding,
            d_model,
            num_heads,
            d_feedforward,
            num_layers_decoder,
            dropout,
        )

        if use_asymmetric_critic:
            self.privileged_encoder = PrivilegedEncoder(
                self.token_embedding,
                self.board_position_embedding,
                d_model,
                num_heads,
                d_feedforward,
                num_layers_critic,
                dropout,
            )
            value_input_dim = d_model * 2
        else:
            self.privileged_encoder = None
            value_input_dim = d_model

        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, value_input_dim // 2),
            nn.ReLU(),
            nn.Linear(value_input_dim // 2, 1),
        )
        # 数量も合法なcategorical候補として扱い、log_probへ含める。

    def encode(
        self, index: torch.Tensor, value: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """公開盤面を符号化する。

        Args:
            index: 語彙index。
            value: indexごとの重み。
            offset: 各tokenの開始位置。

        Returns:
            CLSを含む公開Encoder出力。
        """
        return self.encoder(index, value, offset)

    def encode_privileged(
        self,
        index: torch.Tensor,
        value: torch.Tensor,
        offset: torch.Tensor,
        position_ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """非対称critic用の非公開情報を符号化する。

        Args:
            index: 語彙index。
            value: indexごとの重み。
            offset: 各tokenの開始位置。
            position_ids: unitの盤面位置。
            key_padding_mask: 存在しないhandのmask。

        Returns:
            CLSを含む非公開Encoder出力。

        Raises:
            ValueError: 非対称criticが無効な場合。
        """
        if self.privileged_encoder is None:
            raise ValueError("use_asymmetric_critic=False")
        return self.privileged_encoder(index, value, offset, position_ids, key_padding_mask)

    def value(
        self, encoder_out: torch.Tensor, privileged_out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """公開CLSと、設定時は非公開CLSから価値を計算する。

        Args:
            encoder_out: 公開Encoder出力。
            privileged_out: 非対称criticの非公開Encoder出力。

        Returns:
            形状(batch, 1)の価値。

        Raises:
            ValueError: critic設定とprivileged_outの有無が合わない場合。
        """
        cls_out = encoder_out[:, 0]
        if self.uses_asymmetric_critic:
            if privileged_out is None:
                raise ValueError("use_asymmetric_critic=True requires privileged_out")
            cls_out = torch.cat([cls_out, privileged_out[:, 0]], dim=-1)
        elif privileged_out is not None:
            raise ValueError("use_asymmetric_critic=False but privileged_out was given")
        return self.value_head(cls_out)
