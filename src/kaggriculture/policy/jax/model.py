"""全unitと市場slotを一度に処理するFlax Actor/Critic。"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import actions as A
from kaggriculture.policy.jax.network import Encoder, PrivilegedEncoder, TokenEmbedding
from kaggriculture.rules import constants as C

N_UNIT_SLOTS = C.MAX_HANDS + 1
N_MARKET_SLOTS = C.MAX_MARKET_ORDERS
N_QUERY_SLOTS = N_UNIT_SLOTS + N_MARKET_SLOTS
N_UNIT_ACTIONS = A.UNIT_CANDIDATES.op.shape[0]
N_MARKET_ACTIONS = A.MARKET_CANDIDATES.op.shape[0]
N_QUANTITIES = A.QUANTITY_INDEX.shape[0]


class QueryBlock(nn.Module):
    """queryだけを混合し、固定された状態memoryをcross-attentionで参照する。"""

    config: ModelConfig

    @nn.compact
    def __call__(self, query, memory, active, *, deterministic: bool):
        cfg = self.config
        query_mask = active[:, None, None, :]
        normalized = nn.LayerNorm(name="norm1")(query)
        mixed = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            qkv_features=cfg.d_model,
            dropout_rate=0.0,
            name="self_attn",
        )(normalized, mask=query_mask, deterministic=deterministic)
        query = query + mixed
        normalized = nn.LayerNorm(name="norm2")(query)
        context = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            qkv_features=cfg.d_model,
            dropout_rate=0.0,
            name="cross_attn",
        )(normalized, inputs_k=memory, inputs_v=memory, deterministic=deterministic)
        query = query + context
        normalized = nn.LayerNorm(name="norm3")(query)
        hidden = nn.Dense(cfg.d_feedforward, name="linear1")(normalized)
        hidden = nn.gelu(hidden)
        hidden = nn.Dense(cfg.d_model, name="linear2")(hidden)
        return query + hidden


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
        self.privileged_encoder = PrivilegedEncoder(cfg, name="privileged_encoder")
        self.critic_macro_encoder = nn.Sequential(
            [nn.Dense(cfg.d_model), nn.gelu, nn.Dense(cfg.d_model)],
            name="critic_macro_encoder",
        )
        self.unit_head = nn.Dense(N_UNIT_ACTIONS, name="unit_head")
        self.market_head = nn.Dense(N_MARKET_ACTIONS, name="market_head")
        self.unit_action_embedding = nn.Embed(
            N_UNIT_ACTIONS, cfg.d_model, name="unit_action_embedding"
        )
        self.market_action_embedding = nn.Embed(
            N_MARKET_ACTIONS, cfg.d_model, name="market_action_embedding"
        )
        self.unit_quantity_condition = nn.Dense(cfg.d_model, name="unit_quantity_condition")
        self.market_quantity_condition = nn.Dense(cfg.d_model, name="market_quantity_condition")
        self.quantity_head = nn.Dense(N_QUANTITIES, name="quantity_head")
        self.value_head = nn.Sequential(
            [nn.Dense(cfg.d_model), nn.gelu, nn.Dense(1)], name="value_head"
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
        critic_macro: jnp.ndarray | None = None,
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
        units = queries[:, :N_UNIT_SLOTS]
        market = queries[:, N_UNIT_SLOTS:]
        _ = self.unit_head(units)
        _ = self.market_head(market)
        unit_selected = self.unit_action_embedding(jnp.zeros(units.shape[:-1], jnp.int32))
        market_selected = self.market_action_embedding(jnp.zeros(market.shape[:-1], jnp.int32))
        _ = self.quantity_head(
            self.unit_quantity_condition(jnp.concatenate([units, unit_selected], -1))
        )
        _ = self.quantity_head(
            self.market_quantity_condition(jnp.concatenate([market, market_selected], -1))
        )
        public = jnp.concatenate([memory[:, 0], jnp.mean(memory[:, 1:], axis=1)], axis=-1)
        privileged_embedded = self.token_embedding(privileged_index, privileged_value)
        privileged = self.privileged_encoder(
            privileged_embedded,
            privileged_positions,
            privileged_padding,
            self.board_position_embedding,
            deterministic=deterministic,
        )
        macro = self.critic_macro_encoder(critic_macro)
        value_input = jnp.concatenate([public, privileged[:, 0], macro], axis=-1)
        return queries, self.value_head(value_input)[:, 0]

    def unit_logits(self, hidden, mask):
        """Score the fixed unit action vocabulary."""
        return jnp.where(mask, self.unit_head(hidden), -1e9)

    def market_logits(self, hidden, mask):
        """Score the fixed market action vocabulary."""
        return jnp.where(mask, self.market_head(hidden), -1e9)

    def unit_quantity_logits(self, hidden, action, mask):
        """Score quantities conditioned on a selected unit action."""
        selected = self.unit_action_embedding(action)
        hidden = self.unit_quantity_condition(jnp.concatenate([hidden, selected], -1))
        return jnp.where(mask, self.quantity_head(hidden), -1e9)

    def market_quantity_logits(self, hidden, action, mask):
        """Score quantities conditioned on a selected market action."""
        selected = self.market_action_embedding(action)
        hidden = self.market_quantity_condition(jnp.concatenate([hidden, selected], -1))
        return jnp.where(mask, self.quantity_head(hidden), -1e9)
