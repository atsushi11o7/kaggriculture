"""PyTorch implementation of the fixed-slot fixed-slot policy."""

from __future__ import annotations

import torch
import torch.nn as nn

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.torch.network import Encoder, PrivilegedEncoder, TokenEmbedding
from kaggriculture.rules import constants as C

N_UNIT_SLOTS = C.MAX_HANDS + 1
N_MARKET_SLOTS = C.MAX_MARKET_ORDERS
N_QUERY_SLOTS = N_UNIT_SLOTS + N_MARKET_SLOTS


class QueryBlock(nn.Module):
    """Mix queries, attend to the fixed state memory, then apply an FFN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.d_model, config.num_heads, config.dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            config.d_model, config.num_heads, config.dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.linear1 = nn.Linear(config.d_model, config.d_feedforward)
        self.linear2 = nn.Linear(config.d_feedforward, config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor, active: torch.Tensor
    ) -> torch.Tensor:
        mixed, _ = self.self_attn(query, query, query, key_padding_mask=~active, need_weights=False)
        query = self.norm1(query + mixed)
        context, _ = self.cross_attn(query, memory, memory, need_weights=False)
        query = self.norm2(query + context)
        hidden = self.linear2(torch.relu(self.linear1(query)))
        return self.norm3(query + hidden)


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

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.uses_episode_history = config.use_episode_history
        self.uses_asymmetric_critic = config.use_asymmetric_critic
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
            config.dropout,
        )
        self.query_encoder = ParallelQueryEncoder(self.board_position_embedding, config)
        self.policy_proj = nn.Linear(config.d_model, config.d_model)
        self.quantity_condition = nn.Linear(config.d_model * 2, config.d_model)
        if config.use_asymmetric_critic:
            self.privileged_encoder = PrivilegedEncoder(
                self.token_embedding,
                self.board_position_embedding,
                config.d_model,
                config.num_heads,
                config.d_feedforward,
                config.num_layers_critic,
                config.dropout,
            )
        else:
            self.privileged_encoder = None
        value_width = config.d_model * (2 if config.use_asymmetric_critic else 1)
        self.value_head = nn.Sequential(
            nn.Linear(value_width, value_width // 2), nn.ReLU(), nn.Linear(value_width // 2, 1)
        )

    def embed_dense(self, index: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Embed fixed-width sparse features without constructing EmbeddingBag offsets."""
        embedded = self.token_embedding.bag.weight[index]
        return self.token_embedding.norm((embedded * value[..., None]).sum(dim=-2))

    def forward(
        self,
        encoder_index: torch.Tensor,
        encoder_value: torch.Tensor,
        unit_positions: torch.Tensor,
        unit_active: torch.Tensor,
        unit_inventory_index: torch.Tensor,
        unit_inventory_value: torch.Tensor,
        privileged_index: torch.Tensor | None = None,
        privileged_value: torch.Tensor | None = None,
        privileged_positions: torch.Tensor | None = None,
        privileged_padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed_dense(encoder_index, encoder_value)
        memory = self.encoder.forward_embedded(embedded)
        unit_inventory_embedding = self.embed_dense(unit_inventory_index, unit_inventory_value)
        queries = self.query_encoder(memory, unit_positions, unit_active, unit_inventory_embedding)
        value_input = memory[:, 0]
        if self.uses_asymmetric_critic:
            if any(
                item is None
                for item in (
                    privileged_index,
                    privileged_value,
                    privileged_positions,
                    privileged_padding,
                )
            ):
                raise ValueError("asymmetric critic requires privileged inputs")
            privileged_embedded = self.embed_dense(privileged_index, privileged_value)
            pbatch = privileged_embedded.shape[0]
            pcls = self.privileged_encoder.cls_token.expand(pbatch, 1, -1)
            privileged = torch.cat([pcls, privileged_embedded], dim=1)
            cls_position = torch.full(
                (pbatch, 1), L.NO_POSITION, dtype=torch.long, device=privileged_embedded.device
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
                    torch.zeros((pbatch, 1), dtype=torch.bool, device=privileged_embedded.device),
                    privileged_padding,
                ],
                dim=1,
            )
            privileged = self.privileged_encoder.transformer(
                privileged, src_key_padding_mask=padding
            )
            value_input = torch.cat([value_input, privileged[:, 0]], dim=-1)
        return queries, self.value_head(value_input)[:, 0]

    def score_candidates(
        self,
        hidden: torch.Tensor,
        candidate_index: torch.Tensor,
        candidate_value: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        candidates = self.embed_dense(candidate_index, candidate_value)
        scores = torch.einsum("...d,...cd->...c", self.policy_proj(hidden), candidates)
        # log_softmaxへ-infを流すと、マスクしていない要素の勾配までNaNになる
        # (JAX側と同じ問題。実測で確認済み)。forward値を変えない範囲で
        # 十分小さい有限値を使う。
        return (scores / self.config.d_model**0.5).masked_fill(~candidate_mask, -1e9)

    def condition_quantity(
        self,
        hidden: torch.Tensor,
        selected_index: torch.Tensor,
        selected_value: torch.Tensor,
    ) -> torch.Tensor:
        selected = self.embed_dense(selected_index, selected_value)
        return self.quantity_condition(torch.cat([hidden, selected], dim=-1))
