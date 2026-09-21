"""既存Actorを保ったまま公開Encoderも独立させた非対称critic。"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.jax.model import ParallelQueryEncoder, PolicyValueNet
from kaggriculture.policy.jax.network import Encoder, PrivilegedEncoder, TokenEmbedding

SEPARATED_CRITIC_ARCHITECTURE_VERSION = 1


class SeparatedPolicyValueNet(PolicyValueNet):
    """Actorとcriticがパラメータを一切共有しないPolicyValueNet。

    Actor側のモジュール名と計算は既存のPolicyValueNetと同一に保つ。criticは専用の
    TokenEmbedding、位置embedding、公開Encoder、privileged Encoderを持つため、PPOで
    一方を更新しても他方の中間表現は変化しない。
    """

    def setup(self) -> None:
        cfg = self.config

        # Actor。既存checkpointと同じ名前を維持する。
        self.token_embedding = TokenEmbedding(cfg.d_model, name="token_embedding")
        self.board_position_embedding = self.param(
            "board_position_embedding",
            nn.initializers.normal(0.02),
            (L.N_POSITIONS, cfg.d_model),
        )
        self.encoder = Encoder(cfg, name="encoder")
        self.query_encoder = ParallelQueryEncoder(cfg, name="query_encoder")
        self.policy_proj = nn.Dense(cfg.d_model, name="policy_proj")
        self.quantity_condition = nn.Dense(cfg.d_model, name="quantity_condition")

        # Critic。公開状態を含め、Actorとパラメータを共有しない。
        self.critic_token_embedding = TokenEmbedding(cfg.d_model, name="critic_token_embedding")
        self.critic_board_position_embedding = self.param(
            "critic_board_position_embedding",
            nn.initializers.normal(0.02),
            (L.N_POSITIONS, cfg.d_model),
        )
        self.critic_encoder = Encoder(cfg, name="critic_encoder")
        self.privileged_encoder = PrivilegedEncoder(cfg, name="privileged_encoder")
        self.critic_macro_encoder = nn.Sequential(
            [nn.Dense(cfg.d_model), nn.relu, nn.Dense(cfg.d_model)],
            name="critic_macro_encoder",
        )
        value_width = cfg.d_model * 3
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
        critic_macro: jnp.ndarray | None = None,
        *,
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Actorと独立criticを同じ入力から評価する。"""
        if not self.config.use_asymmetric_critic:
            raise ValueError("separated critic requires use_asymmetric_critic=true")
        if any(
            value is None
            for value in (
                privileged_index,
                privileged_value,
                privileged_positions,
                privileged_padding,
                critic_macro,
            )
        ):
            raise ValueError("separated critic requires privileged inputs")

        actor_embedded = self.token_embedding(encoder_index, encoder_value)
        actor_memory = self.encoder(
            actor_embedded,
            self.board_position_embedding,
            deterministic=deterministic,
        )
        unit_inventory_embedding = self.token_embedding(unit_inventory_index, unit_inventory_value)
        queries = self.query_encoder(
            actor_memory,
            unit_positions,
            unit_active,
            self.board_position_embedding,
            unit_inventory_embedding,
            deterministic=deterministic,
        )
        _ = self.policy_proj(queries)
        _ = self.quantity_condition(jnp.concatenate([queries, queries], axis=-1))

        critic_embedded = self.critic_token_embedding(encoder_index, encoder_value)
        critic_memory = self.critic_encoder(
            critic_embedded,
            self.critic_board_position_embedding,
            deterministic=deterministic,
        )
        privileged_embedded = self.critic_token_embedding(privileged_index, privileged_value)
        privileged = self.privileged_encoder(
            privileged_embedded,
            privileged_positions,
            privileged_padding,
            self.critic_board_position_embedding,
            deterministic=deterministic,
        )
        macro = self.critic_macro_encoder(critic_macro)
        value_input = jnp.concatenate([critic_memory[:, 0], privileged[:, 0], macro], axis=-1)
        return queries, self.value_head(value_input)[:, 0]
