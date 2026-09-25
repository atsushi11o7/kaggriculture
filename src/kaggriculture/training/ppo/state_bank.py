"""Pre-generated day<=11 handoff states, for PPO rollouts that start at day 12.

Every PPO rollout iteration otherwise pays the cost of simulating the fixed,
non-learned day<=11 opening from scratch before any of the trajectory
becomes eligible for policy loss. Since that opening is deterministic-ish
(rule-based or a frozen BC checkpoint, not the policy under training), it's
cheaper to generate a bank of realistic day-12 starting states once, offline,
and have rollouts sample from it instead of re-simulating day<=11 every time.

See src/kaggriculture/training/opening/generate_state_bank.py (or the
scratch script that produced a given bank file) for how a bank is built:
each entry is a ``State`` snapshot at step 288 (day 12, hour 0), pickled one
after another into a single file.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp

from kaggriculture.simulator.state import State


def load_state_bank(path: str | Path) -> State:
    """Load a pickled sequence of single-game States into one batched State."""
    states = []
    with open(path, "rb") as f:
        while True:
            try:
                states.append(pickle.load(f))
            except EOFError:
                break
    if not states:
        raise ValueError(f"state bank at {path} is empty")
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    return stacked


def make_bank_reset_fn(bank: State) -> Callable[..., State]:
    """Build a ``reset``-compatible sampler that draws (with replacement) from ``bank``.

    Matches the call signature of :func:`kaggriculture.simulator.reset.reset`
    (``key, batch_size, board_size=..., starting_money=...``) so it can be
    swapped in wherever ``reset`` is currently called, including inside
    ``collect_rollout``'s jitted scan body: it's pure array indexing, so it's
    jit/vmap-safe.
    """
    bank_size = jax.tree.util.tree_leaves(bank)[0].shape[0]

    def bank_reset(key, batch_size, board_size=None, starting_money=None):
        del board_size, starting_money
        indices = jax.random.randint(key, (batch_size,), 0, bank_size)
        return jax.tree.map(lambda x: x[indices], bank)

    return bank_reset
