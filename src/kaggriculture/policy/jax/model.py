"""全unitと市場slotを一度に処理するFlax Actor/Critic。"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax.network import Encoder, PrivilegedEncoder, TokenEmbedding
from kaggriculture.rules import constants as C

N_UNIT_SLOTS = C.MAX_HANDS + 1
N_MARKET_SLOTS = C.MAX_MARKET_ORDERS
N_QUERY_SLOTS = N_UNIT_SLOTS + N_MARKET_SLOTS


class QueryBlock(nn.Module):
    """queryだけを混合し、固定された状態memoryをcross-attentionで参照する。"""

    config: ModelConfig

    @nn.compact
    def __call__(self, query, memory, active, *, deterministic: bool):
        cfg = self.config
        query_mask = active[:, None, None, :]
        mixed = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            qkv_features=cfg.d_model,
            dropout_rate=cfg.dropout,
            name="self_attn",
        )(query, mask=query_mask, deterministic=deterministic)
        query = nn.LayerNorm(name="norm1")(query + mixed)
        context = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            qkv_features=cfg.d_model,
            dropout_rate=cfg.dropout,
            name="cross_attn",
        )(query, inputs_k=memory, inputs_v=memory, deterministic=deterministic)
        query = nn.LayerNorm(name="norm2")(query + context)
        hidden = nn.Dense(cfg.d_feedforward, name="linear1")(query)
        hidden = nn.relu(hidden)
        hidden = nn.Dense(cfg.d_model, name="linear2")(hidden)
        return nn.LayerNorm(name="norm3")(query + hidden)


class ParallelQueryEncoder(nn.Module):
    """全queryを共同処理し、状態memoryは更新せずcross-attentionする。"""

    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        memory,
        unit_positions,
        unit_active,
        board_position_embedding,
        unit_inventory_embedding,
        *,
        deterministic,
    ):
        cfg = self.config
        batch = memory.shape[0]
        slot = nn.Embed(N_QUERY_SLOTS, cfg.d_model, name="slot_embedding")(
            jnp.arange(N_QUERY_SLOTS)
        )
        kinds = jnp.concatenate(
            [jnp.zeros(N_UNIT_SLOTS, jnp.int32), jnp.ones(N_MARKET_SLOTS, jnp.int32)]
        )
        kind = nn.Embed(2, cfg.d_model, name="kind_embedding")(kinds)
        positions = jnp.concatenate(
            [unit_positions, jnp.full((batch, N_MARKET_SLOTS), L.NO_POSITION)], axis=1
        )
        query = slot[None] + kind[None] + board_position_embedding[positions]
        # unit slot(farmer+hands)だけ、そのunit自身が運んでいるinventoryの
        # embeddingを加算する(market slotには対象unitが無いので加算しない)。
        # これが無いと、方策はどのunitが何を持っているかを一切区別できず、
        # PLACE/DROP/SELL等でどのunitに行わせるべきかを判断できない。
        unit_query = query[:, :N_UNIT_SLOTS] + unit_inventory_embedding
        query = jnp.concatenate([unit_query, query[:, N_UNIT_SLOTS:]], axis=1)
        active = jnp.concatenate([unit_active, jnp.ones((batch, N_MARKET_SLOTS), bool)], axis=1)
        for layer in range(cfg.num_layers_decoder):
            query = QueryBlock(cfg, name=f"layer_{layer}")(
                query, memory, active, deterministic=deterministic
            )
        return query


class PolicyValueNet(nn.Module):
    """固定slotの候補を一括採点する非自己回帰Actor/Critic。"""

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
        self.query_encoder = ParallelQueryEncoder(cfg, name="query_encoder")
        self.privileged_encoder = (
            PrivilegedEncoder(cfg, name="privileged_encoder") if cfg.use_asymmetric_critic else None
        )
        self.policy_proj = nn.Dense(cfg.d_model, name="policy_proj")
        self.quantity_condition = nn.Dense(cfg.d_model, name="quantity_condition")
        value_width = cfg.d_model * (2 if cfg.use_asymmetric_critic else 1)
        self.value_head = nn.Sequential(
            [nn.Dense(value_width // 2), nn.relu, nn.Dense(1)], name="value_head"
        )

    def __call__(
        self,
        encoder_index: jnp.ndarray,
        encoder_value: jnp.ndarray,
        unit_positions: jnp.ndarray,
        unit_active: jnp.ndarray,
        unit_inventory_index: jnp.ndarray,
        unit_inventory_value: jnp.ndarray,
        privileged_index: jnp.ndarray | None = None,
        privileged_value: jnp.ndarray | None = None,
        privileged_positions: jnp.ndarray | None = None,
        privileged_padding: jnp.ndarray | None = None,
        *,
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """状態を1回符号化し、全slot表現と価値を返す。"""
        embedded = self.token_embedding(encoder_index, encoder_value)
        memory = self.encoder(
            embedded,
            self.board_position_embedding,
            deterministic=deterministic,
        )
        unit_inventory_embedding = self.token_embedding(unit_inventory_index, unit_inventory_value)
        queries = self.query_encoder(
            memory,
            unit_positions,
            unit_active,
            self.board_position_embedding,
            unit_inventory_embedding,
            deterministic=deterministic,
        )
        # Head parameterも同じinit呼び出しで生成する。実際の採点は公開methodで行う。
        _ = self.policy_proj(queries)
        _ = self.quantity_condition(jnp.concatenate([queries, queries], axis=-1))
        value_input = memory[:, 0]
        if self.config.use_asymmetric_critic:
            if any(
                value is None
                for value in (
                    privileged_index,
                    privileged_value,
                    privileged_positions,
                    privileged_padding,
                )
            ):
                raise ValueError("asymmetric critic requires privileged inputs")
            privileged_embedded = self.token_embedding(privileged_index, privileged_value)
            privileged = self.privileged_encoder(
                privileged_embedded,
                privileged_positions,
                privileged_padding,
                self.board_position_embedding,
                deterministic=deterministic,
            )
            value_input = jnp.concatenate([value_input, privileged[:, 0]], axis=-1)
        return queries, self.value_head(value_input)[:, 0]

    def score_candidates(
        self,
        hidden: jnp.ndarray,
        candidate_index: jnp.ndarray,
        candidate_value: jnp.ndarray,
        candidate_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """任意の先行shapeを保ったまま候補を内積採点する。"""
        candidates = self.token_embedding(candidate_index, candidate_value)
        scores = jnp.einsum("...d,...cd->...c", self.policy_proj(hidden), candidates)
        scores = scores / jnp.sqrt(jnp.asarray(self.config.d_model, dtype=scores.dtype))
        # log_softmaxへ-infを流すと、マスクしていない要素の勾配までNaNになる
        # (JAXでの既知の問題)。forward値を変えない範囲で十分小さい有限値を使う。
        return jnp.where(candidate_mask, scores, -1e9)

    def condition_quantity(
        self,
        hidden: jnp.ndarray,
        selected_index: jnp.ndarray,
        selected_value: jnp.ndarray,
    ) -> jnp.ndarray:
        """選択したop/itemで数量queryを条件付けする。"""
        selected = self.token_embedding(selected_index, selected_value)
        return self.quantity_condition(jnp.concatenate([hidden, selected], axis=-1))
