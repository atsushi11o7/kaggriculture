"""並列方策の固定shape入出力。"""

from typing import NamedTuple

import jax.numpy as jnp

from kaggriculture.simulator.action import Action


class Intent(NamedTuple):
    """Executor適用前に方策が選んだ候補index。"""

    unit: jnp.ndarray
    unit_quantity: jnp.ndarray
    market: jnp.ndarray
    market_quantity: jnp.ndarray


class ExecutorStats(NamedTuple):
    """方策出力を合法化した回数。"""

    invalid_unit: jnp.ndarray
    clamped_unit_quantity: jnp.ndarray
    blocked_plants: jnp.ndarray
    invalid_market: jnp.ndarray
    clamped_market_quantity: jnp.ndarray
    ignored_after_stop: jnp.ndarray


class PolicyOutput(NamedTuple):
    """サンプリング結果とPPOに必要な統計。"""

    intent: Intent
    action: Action
    log_prob: jnp.ndarray
    slot_log_prob: jnp.ndarray
    slot_mask: jnp.ndarray
    entropy: jnp.ndarray
    value: jnp.ndarray
    stats: ExecutorStats


class EvaluationOutput(NamedTuple):
    """保存済みintentを現在の方策で再評価した結果。"""

    log_prob: jnp.ndarray
    slot_log_prob: jnp.ndarray
    slot_mask: jnp.ndarray
    slot_valid: jnp.ndarray
    entropy: jnp.ndarray
    value: jnp.ndarray
