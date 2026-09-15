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
