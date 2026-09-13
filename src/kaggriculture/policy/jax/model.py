"""Flaxによる学習用JAX Actor/Critic。

提出用のPyTorchモデルと同じpost-norm Transformer構造を持つ。入力は
`jax_features`で固定形状化した(index, value)で、paddingはvalue=0として扱う。
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import features as F


class TokenEmbedding(nn.Module):
    """固定幅の疎特徴量を重み付き加算し、LayerNormする。"""

    d_model: int

    @nn.compact
    def __call__(self, index: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
        embedding = self.param("embedding", nn.initializers.normal(), (V.VOCAB_SIZE, self.d_model))
        x = jnp.sum(embedding[index] * value[..., None], axis=-2)
        return nn.LayerNorm(epsilon=1e-5, name="norm")(x)


class EncoderBlock(nn.Module):
    """PyTorch TransformerEncoderLayer互換のpost-normブロック。"""

    config: ModelConfig

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, *, mask: jnp.ndarray | None, deterministic: bool
    ) -> jnp.ndarray:
        cfg = self.config
        attention = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            qkv_features=cfg.d_model,
            out_features=cfg.d_model,
            dropout_rate=cfg.dropout,
            use_bias=True,
            name="self_attn",
        )(x, mask=mask, deterministic=deterministic)
        x = nn.LayerNorm(epsilon=1e-5, name="norm1")(
            x + nn.Dropout(cfg.dropout, name="dropout1")(attention, deterministic=deterministic)
        )
        hidden = nn.Dense(cfg.d_feedforward, name="linear1")(x)
        hidden = nn.relu(hidden)
        hidden = nn.Dropout(cfg.dropout, name="dropout")(hidden, deterministic=deterministic)
        hidden = nn.Dense(cfg.d_model, name="linear2")(hidden)
        return nn.LayerNorm(epsilon=1e-5, name="norm2")(
            x + nn.Dropout(cfg.dropout, name="dropout2")(hidden, deterministic=deterministic)
        )


class DecoderBlock(nn.Module):
    """PyTorch TransformerDecoderLayer互換のpost-normブロック。"""

    config: ModelConfig
    decode: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        memory: jnp.ndarray,
        *,
        self_mask: jnp.ndarray,
        deterministic: bool,
    ) -> jnp.ndarray:
        cfg = self.config
        attention = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            qkv_features=cfg.d_model,
            out_features=cfg.d_model,
            dropout_rate=cfg.dropout,
            use_bias=True,
            decode=self.decode,
            name="self_attn",
        )(x, mask=self_mask, deterministic=deterministic)
        x = nn.LayerNorm(epsilon=1e-5, name="norm1")(
            x + nn.Dropout(cfg.dropout, name="dropout1")(attention, deterministic=deterministic)
        )
        attention = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            qkv_features=cfg.d_model,
            out_features=cfg.d_model,
            dropout_rate=cfg.dropout,
            use_bias=True,
            name="multihead_attn",
        )(x, inputs_k=memory, inputs_v=memory, deterministic=deterministic)
        x = nn.LayerNorm(epsilon=1e-5, name="norm2")(
            x + nn.Dropout(cfg.dropout, name="dropout2")(attention, deterministic=deterministic)
        )
        hidden = nn.Dense(cfg.d_feedforward, name="linear1")(x)
        hidden = nn.relu(hidden)
        hidden = nn.Dropout(cfg.dropout, name="dropout")(hidden, deterministic=deterministic)
        hidden = nn.Dense(cfg.d_model, name="linear2")(hidden)
        return nn.LayerNorm(epsilon=1e-5, name="norm3")(
            x + nn.Dropout(cfg.dropout, name="dropout3")(hidden, deterministic=deterministic)
        )


class Encoder(nn.Module):
    """公開観測のTransformer Encoder。"""

    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        embedded: jnp.ndarray,
        board_position_embedding: jnp.ndarray,
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        cfg = self.config
        batch_size = embedded.shape[0]
        cls = self.param("cls_token", nn.initializers.zeros, (1, 1, cfg.d_model))
        x = jnp.concatenate([jnp.broadcast_to(cls, (batch_size, 1, cfg.d_model)), embedded], axis=1)
        owner_ids, zone_ids, position_ids = zip(*L.TOKEN_OWNER_ZONE_POSITION_WITH_CLS, strict=True)
        owner = nn.Embed(
            L.N_OWNERS,
            cfg.d_model,
            embedding_init=nn.initializers.normal(0.02),
            name="owner_embedding",
        )(jnp.asarray(owner_ids))
        zone = nn.Embed(
            L.N_ZONES,
            cfg.d_model,
            embedding_init=nn.initializers.normal(0.02),
            name="zone_embedding",
        )(jnp.asarray(zone_ids))
        x = x + owner + zone + board_position_embedding[jnp.asarray(position_ids)]
        for layer in range(cfg.num_layers_encoder):
            x = EncoderBlock(cfg, name=f"layer_{layer}")(x, mask=None, deterministic=deterministic)
        return x


class Decoder(nn.Module):
    """決定済みprefixと公開memoryから次の各スロットを符号化する。"""

    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        embedded: jnp.ndarray,
        memory: jnp.ndarray,
        position_ids: jnp.ndarray,
        board_position_ids: jnp.ndarray,
        padding_mask: jnp.ndarray,
        board_position_embedding: jnp.ndarray,
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        cfg = self.config
        position_embedding = nn.Embed(
            L.MAX_DECODE_LEN,
            cfg.d_model,
            embedding_init=nn.initializers.normal(0.02),
            name="position_embedding",
        )(position_ids)
        x = embedded + position_embedding + board_position_embedding[board_position_ids]
        seq_len = x.shape[1]
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        key_valid = jnp.logical_not(padding_mask)[:, None, None, :]
        self_mask = causal[None, None, :, :] & key_valid
        for layer in range(cfg.num_layers_decoder):
            x = DecoderBlock(cfg, name=f"layer_{layer}")(
                x, memory, self_mask=self_mask, deterministic=deterministic
            )
        return x

    @nn.compact
    def decode_step(
        self,
        embedded: jnp.ndarray,
        memory: jnp.ndarray,
        position_ids: jnp.ndarray,
        board_position_ids: jnp.ndarray,
        board_position_embedding: jnp.ndarray,
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        """1スロットだけ処理し、各self-attention層のKV cacheを更新する。"""
        cfg = self.config
        position_embedding = nn.Embed(
            L.MAX_DECODE_LEN,
            cfg.d_model,
            embedding_init=nn.initializers.normal(0.02),
            name="position_embedding",
        )(position_ids)
        x = embedded + position_embedding + board_position_embedding[board_position_ids]
        for layer in range(cfg.num_layers_decoder):
            x = DecoderBlock(cfg, decode=True, name=f"layer_{layer}")(
                x, memory, self_mask=None, deterministic=deterministic
            )
        return x


class PrivilegedEncoder(nn.Module):
    """非対称critic専用の非公開情報Encoder。"""

    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        embedded: jnp.ndarray,
        position_ids: jnp.ndarray,
        padding_mask: jnp.ndarray,
        board_position_embedding: jnp.ndarray,
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        cfg = self.config
        batch_size = embedded.shape[0]
        cls = self.param("cls_token", nn.initializers.zeros, (1, 1, cfg.d_model))
        x = jnp.concatenate([jnp.broadcast_to(cls, (batch_size, 1, cfg.d_model)), embedded], axis=1)
        owner_ids, zone_ids = zip(*L.PRIVILEGED_OWNER_ZONE_WITH_CLS, strict=True)
        owner = nn.Embed(
            L.N_OWNERS,
            cfg.d_model,
            embedding_init=nn.initializers.normal(0.02),
            name="owner_embedding",
        )(jnp.asarray(owner_ids))
        zone = nn.Embed(
            L.N_ZONES,
            cfg.d_model,
            embedding_init=nn.initializers.normal(0.02),
            name="zone_embedding",
        )(jnp.asarray(zone_ids))
        cls_position = jnp.full((batch_size, 1), L.NO_POSITION, dtype=position_ids.dtype)
        all_positions = jnp.concatenate([cls_position, position_ids], axis=1)
        x = x + owner + zone + board_position_embedding[all_positions]
        all_padding = jnp.concatenate(
            [jnp.zeros((batch_size, 1), dtype=bool), padding_mask], axis=1
        )
        key_valid = jnp.logical_not(all_padding)[:, None, None, :]
        for layer in range(cfg.num_layers_critic):
            x = EncoderBlock(cfg, name=f"layer_{layer}")(
                x, mask=key_valid, deterministic=deterministic
            )
        return x


class PolicyValueNet(nn.Module):
    """学習用JAX方策・価値ネットワーク。"""

    config: ModelConfig

    def setup(self) -> None:
        cfg = self.config
        self.token_embedding = TokenEmbedding(cfg.d_model, name="token_embedding")
        self.board_position_embedding = self.param(
            "board_position_embedding",
            nn.initializers.normal(0.02),
            (L.N_POSITIONS, cfg.d_model),
        )
        self.encoder = Encoder(cfg, name="encoder")
        self.decoder = Decoder(cfg, name="decoder")
        self.policy_proj = nn.Dense(cfg.d_model, name="policy_proj")
        if cfg.use_asymmetric_critic:
            self.privileged_encoder = PrivilegedEncoder(cfg, name="privileged_encoder")
        self.value_hidden = nn.Dense(
            cfg.d_model if cfg.use_asymmetric_critic else cfg.d_model // 2,
            name="value_hidden",
        )
        self.value_out = nn.Dense(1, name="value_out")

    def encode(
        self, index: jnp.ndarray, value: jnp.ndarray, *, deterministic: bool = True
    ) -> jnp.ndarray:
        embedded = self.token_embedding(index, value)
        return self.encoder(embedded, self.board_position_embedding, deterministic=deterministic)

    def decode(
        self,
        memory: jnp.ndarray,
        index: jnp.ndarray,
        value: jnp.ndarray,
        position_ids: jnp.ndarray,
        board_position_ids: jnp.ndarray,
        padding_mask: jnp.ndarray,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        embedded = self.token_embedding(index, value)
        return self.decoder(
            embedded,
            memory,
            position_ids,
            board_position_ids,
            padding_mask,
            self.board_position_embedding,
            deterministic=deterministic,
        )

    def decode_step(
        self,
        memory: jnp.ndarray,
        index: jnp.ndarray,
        value: jnp.ndarray,
        position_ids: jnp.ndarray,
        board_position_ids: jnp.ndarray,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """KV cacheを使って1決定スロットを増分デコードする。"""
        embedded = self.token_embedding(index, value)
        return self.decoder.decode_step(
            embedded,
            memory,
            position_ids,
            board_position_ids,
            self.board_position_embedding,
            deterministic=deterministic,
        )

    def score_candidates(
        self,
        hidden: jnp.ndarray,
        candidate_index: jnp.ndarray,
        candidate_value: jnp.ndarray,
        candidate_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        candidates = self.token_embedding(candidate_index, candidate_value)
        projected = self.policy_proj(hidden)
        scores = jnp.einsum("...cd,...d->...c", candidates, projected)
        scores = scores / jnp.sqrt(jnp.asarray(self.config.d_model, dtype=scores.dtype))
        return jnp.where(candidate_mask, scores, -jnp.inf)

    def encode_privileged(
        self,
        index: jnp.ndarray,
        value: jnp.ndarray,
        position_ids: jnp.ndarray,
        padding_mask: jnp.ndarray,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        if not self.config.use_asymmetric_critic:
            raise ValueError("use_asymmetric_critic=False")
        embedded = self.token_embedding(index, value)
        return self.privileged_encoder(
            embedded,
            position_ids,
            padding_mask,
            self.board_position_embedding,
            deterministic=deterministic,
        )

    def get_value(self, memory: jnp.ndarray, privileged: jnp.ndarray | None = None) -> jnp.ndarray:
        x = memory[:, 0]
        if self.config.use_asymmetric_critic:
            if privileged is None:
                raise ValueError("asymmetric critic requires privileged input")
            x = jnp.concatenate([x, privileged[:, 0]], axis=-1)
        elif privileged is not None:
            raise ValueError("symmetric critic does not accept privileged input")
        return self.value_out(nn.relu(self.value_hidden(x))).squeeze(-1)

    def __call__(
        self,
        encoder_index: jnp.ndarray,
        encoder_value: jnp.ndarray,
        decoder_index: jnp.ndarray,
        decoder_value: jnp.ndarray,
        decoder_position_ids: jnp.ndarray,
        decoder_board_position_ids: jnp.ndarray,
        decoder_padding_mask: jnp.ndarray,
        candidate_index: jnp.ndarray,
        candidate_value: jnp.ndarray,
        candidate_mask: jnp.ndarray,
        privileged_index: jnp.ndarray | None = None,
        privileged_value: jnp.ndarray | None = None,
        privileged_position_ids: jnp.ndarray | None = None,
        privileged_padding_mask: jnp.ndarray | None = None,
        *,
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        memory = self.encode(encoder_index, encoder_value, deterministic=deterministic)
        hidden = self.decode(
            memory,
            decoder_index,
            decoder_value,
            decoder_position_ids,
            decoder_board_position_ids,
            decoder_padding_mask,
            deterministic=deterministic,
        )
        scores = self.score_candidates(hidden, candidate_index, candidate_value, candidate_mask)
        privileged = None
        if self.config.use_asymmetric_critic:
            if any(
                item is None
                for item in (
                    privileged_index,
                    privileged_value,
                    privileged_position_ids,
                    privileged_padding_mask,
                )
            ):
                raise ValueError("asymmetric critic inputs are required")
            privileged = self.encode_privileged(
                privileged_index,
                privileged_value,
                privileged_position_ids,
                privileged_padding_mask,
                deterministic=deterministic,
            )
        return scores, self.get_value(memory, privileged)


def initialize(model: PolicyValueNet, key, batch_size: int = 1) -> dict:
    """全parameterをJAX上で初期化する。"""
    encoder_index = jnp.zeros(
        (batch_size, L.NUM_WORDS_ENCODER, F.MAX_ENCODER_FEATURES), dtype=jnp.int32
    )
    encoder_value = jnp.zeros_like(encoder_index, dtype=jnp.float32)
    decoder_index = jnp.zeros((batch_size, 1, F.MAX_DECODER_FEATURES), dtype=jnp.int32)
    decoder_value = jnp.zeros_like(decoder_index, dtype=jnp.float32)
    decoder_positions = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    decoder_board_positions = jnp.full_like(decoder_positions, L.NO_POSITION)
    decoder_padding = jnp.zeros_like(decoder_positions, dtype=bool)
    candidate_index = jnp.zeros((batch_size, 1, 1, F.MAX_CANDIDATE_FEATURES), dtype=jnp.int32)
    candidate_value = jnp.ones_like(candidate_index, dtype=jnp.float32)
    candidate_mask = jnp.ones((batch_size, 1, 1), dtype=bool)

    privileged = {}
    if model.config.use_asymmetric_critic:
        n_privileged = len(L.PRIVILEGED_OWNER_ZONE_WITH_CLS) - 1
        privileged = {
            "privileged_index": jnp.zeros(
                (batch_size, n_privileged, F.MAX_PRIVILEGED_FEATURES), dtype=jnp.int32
            ),
            "privileged_value": jnp.zeros(
                (batch_size, n_privileged, F.MAX_PRIVILEGED_FEATURES), dtype=jnp.float32
            ),
            "privileged_position_ids": jnp.full(
                (batch_size, n_privileged), L.NO_POSITION, dtype=jnp.int32
            ),
            "privileged_padding_mask": jnp.zeros((batch_size, n_privileged), dtype=bool),
        }
    return model.init(
        key,
        encoder_index,
        encoder_value,
        decoder_index,
        decoder_value,
        decoder_positions,
        decoder_board_positions,
        decoder_padding,
        candidate_index,
        candidate_value,
        candidate_mask,
        **privileged,
    )
