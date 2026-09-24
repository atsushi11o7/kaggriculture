"""PyTorch implementation of the fixed-slot policy."""

from __future__ import annotations

import torch
import torch.nn as nn

from kaggriculture.policy.common import action_schema as S
from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common.config import NUM_CRITIC_MACRO_FEATURES, ModelConfig
from kaggriculture.policy.torch.network import Encoder, PrivilegedEncoder, TokenEmbedding
from kaggriculture.rules import constants as C

N_UNIT_SLOTS = C.MAX_HANDS + 1
N_MARKET_SLOTS = C.MAX_MARKET_ORDERS
N_QUERY_SLOTS = N_UNIT_SLOTS + N_MARKET_SLOTS
N_UNIT_ACTIONS = S.N_UNIT_ACTIONS
N_MARKET_ACTIONS = S.N_MARKET_ACTIONS
N_QUANTITIES = S.N_QUANTITIES


class QueryBlock(nn.Module):
    """Mix queries, attend to the fixed state memory, then apply an FFN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.d_model, config.num_heads, 0.0, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            config.d_model, config.num_heads, 0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.linear1 = nn.Linear(config.d_model, config.d_feedforward)
        self.linear2 = nn.Linear(config.d_feedforward, config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor, active: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.norm1(query)
        mixed, _ = self.self_attn(
            normalized, normalized, normalized, key_padding_mask=~active, need_weights=False
        )
        query = query + mixed
        normalized = self.norm2(query)
        context, _ = self.cross_attn(normalized, memory, memory, need_weights=False)
        query = query + context
        hidden = self.linear2(torch.nn.functional.gelu(self.linear1(self.norm3(query))))
        return query + hidden


class ParallelQueryEncoder(nn.Module):
    """Produce all unit and market slot representations in parallel."""

    def __init__(
        self,
        board_position_embedding: nn.Embedding,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.board_position_embedding = board_position_embedding
        self.slot_embedding = nn.Embedding(N_QUERY_SLOTS, config.d_model)
        self.kind_embedding = nn.Embedding(2, config.d_model)
        self.layers = nn.ModuleList(QueryBlock(config) for _ in range(config.num_layers_decoder))

    def forward(
        self,
        memory: torch.Tensor,
        unit_positions: torch.Tensor,
        unit_active: torch.Tensor,
        unit_inventory_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch = memory.shape[0]
        device = memory.device
        slots = torch.arange(N_QUERY_SLOTS, device=device)
        kinds = torch.cat(
            [
                torch.zeros(N_UNIT_SLOTS, dtype=torch.long, device=device),
                torch.ones(N_MARKET_SLOTS, dtype=torch.long, device=device),
            ]
        )
        market_positions = torch.full(
            (batch, N_MARKET_SLOTS), L.NO_POSITION, dtype=torch.long, device=device
        )
        positions = torch.cat([unit_positions, market_positions], dim=1)
        query = (
            self.slot_embedding(slots)[None]
            + self.kind_embedding(kinds)[None]
            + self.board_position_embedding(positions)
        )
        # unit slot(farmer+hands)だけ、そのunit自身が運んでいるinventoryの
        # embeddingを加算する(JAX側のParallelQueryEncoderと同じ)。
        query = torch.cat(
            [query[:, :N_UNIT_SLOTS] + unit_inventory_embedding, query[:, N_UNIT_SLOTS:]], dim=1
        )
        active = torch.cat(
            [unit_active, torch.ones((batch, N_MARKET_SLOTS), dtype=torch.bool, device=device)],
            dim=1,
        )
        for layer in self.layers:
            query = layer(query, memory, active)
        return query


class PolicyValueNet(nn.Module):
    """One-pass actor with an optional asymmetric critic."""

    def __init__(self, config: ModelConfig, *, actor_only: bool = False) -> None:
        super().__init__()
        self.config = config
        self.actor_only = actor_only
        self.token_embedding = TokenEmbedding(config.d_model)
        self.board_position_embedding = nn.Embedding(L.N_POSITIONS, config.d_model)
        nn.init.normal_(self.board_position_embedding.weight, std=0.02)
        self.encoder = Encoder(
            self.token_embedding,
            self.board_position_embedding,
            config.d_model,
            config.num_heads,
            config.d_feedforward,
            config.num_layers_encoder,
            0.0,
        )
        self.query_encoder = ParallelQueryEncoder(self.board_position_embedding, config)
        self.unit_head = nn.Linear(config.d_model, N_UNIT_ACTIONS)
        self.market_head = nn.Linear(config.d_model, N_MARKET_ACTIONS)
        self.unit_action_embedding = nn.Embedding(N_UNIT_ACTIONS, config.d_model)
        self.market_action_embedding = nn.Embedding(N_MARKET_ACTIONS, config.d_model)
        self.unit_quantity_condition = nn.Linear(config.d_model * 2, config.d_model)
        self.market_quantity_condition = nn.Linear(config.d_model * 2, config.d_model)
        self.quantity_head = nn.Linear(config.d_model, N_QUANTITIES)
        self.privileged_encoder = None
        self.critic_macro_encoder = None
        self.value_head = None
        if not actor_only:
            self.privileged_encoder = PrivilegedEncoder(
                self.token_embedding,
                self.board_position_embedding,
                config.d_model,
                config.num_heads,
                config.d_feedforward,
                config.num_layers_critic,
                0.0,
            )
            self.critic_macro_encoder = nn.Sequential(
                nn.Linear(NUM_CRITIC_MACRO_FEATURES, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.value_head = nn.Sequential(
                nn.Linear(config.d_model * 4, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )

    def embed_dense(self, index: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Embed fixed-width sparse features without constructing EmbeddingBag offsets."""
        embedded = self.token_embedding.bag.weight[index]
        return self.token_embedding.norm((embedded * value[..., None]).sum(dim=-2))

    def encode_actor(
        self,
        encoder_index,
        encoder_value,
        unit_positions,
        unit_active,
        unit_inventory_index,
        unit_inventory_value,
    ):
        """Encode public state into all parallel action slots."""
        embedded = self.embed_dense(encoder_index, encoder_value)
        memory = self.encoder.forward_embedded(embedded)
        inventory = self.embed_dense(unit_inventory_index, unit_inventory_value)
        queries = self.query_encoder(memory, unit_positions, unit_active, inventory)
        public = torch.cat([memory[:, 0], memory[:, 1:].mean(dim=1)], dim=-1)
        return queries, public

    def forward(
        self,
        encoder_index,
        encoder_value,
        unit_positions,
        unit_active,
        unit_inventory_index,
        unit_inventory_value,
        privileged_index,
        privileged_value,
        privileged_positions,
        privileged_padding,
        critic_macro,
    ):
        if self.actor_only:
            raise RuntimeError("actor-only submission model has no critic")
        queries, public = self.encode_actor(
            encoder_index,
            encoder_value,
            unit_positions,
            unit_active,
            unit_inventory_index,
            unit_inventory_value,
        )
        privileged_embedded = self.embed_dense(privileged_index, privileged_value)
        batch = privileged_embedded.shape[0]
        cls = self.privileged_encoder.cls_token.expand(batch, 1, -1)
        privileged = torch.cat([cls, privileged_embedded], dim=1)
        cls_position = torch.full(
            (batch, 1), L.NO_POSITION, dtype=torch.long, device=privileged.device
        )
        positions = torch.cat([cls_position, privileged_positions], dim=1)
        privileged = privileged + self.privileged_encoder.owner_embedding(
            self.privileged_encoder._owner_ids
        )
        privileged = privileged + self.privileged_encoder.zone_embedding(
            self.privileged_encoder._zone_ids
        )
        privileged = privileged + self.board_position_embedding(positions)
        padding = torch.cat(
            [
                torch.zeros((batch, 1), dtype=torch.bool, device=privileged.device),
                privileged_padding,
            ],
            dim=1,
        )
        privileged = self.privileged_encoder.transformer(privileged, src_key_padding_mask=padding)
        macro = self.critic_macro_encoder(critic_macro)
        value = self.value_head(torch.cat([public, privileged[:, 0], macro], dim=-1))[:, 0]
        return queries, value

    def unit_logits(self, hidden, mask):
        return self.unit_head(hidden).masked_fill(~mask, -1e9)

    def market_logits(self, hidden, mask):
        return self.market_head(hidden).masked_fill(~mask, -1e9)

    def unit_quantity_logits(self, hidden, action, mask):
        selected = self.unit_action_embedding(action)
        hidden = self.unit_quantity_condition(torch.cat([hidden, selected], dim=-1))
        return self.quantity_head(hidden).masked_fill(~mask, -1e9)

    def market_quantity_logits(self, hidden, action, mask):
        selected = self.market_action_embedding(action)
        hidden = self.market_quantity_condition(torch.cat([hidden, selected], dim=-1))
        return self.quantity_head(hidden).masked_fill(~mask, -1e9)
