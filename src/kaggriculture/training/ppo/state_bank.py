"""Opening handoff states for PPO rollouts from displayed day 12."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture.policy.jax import history as H
from kaggriculture.policy.jax.tokenize import EpisodeCounters
from kaggriculture.simulator.state import State
from kaggriculture.training.rl import estimated_assets


class StateBank(NamedTuple):
    """Batched simulator state and observable history at one handoff step."""

    state: State
    counters: EpisodeCounters


class RolloutStart(NamedTuple):
    """A sampled batch ready to enter a PPO rollout."""

    state: State
    counters: EpisodeCounters
    margin: jax.Array


def _stack_leaves(*values):
    first = values[0]
    if hasattr(first, "dtype") and jax.dtypes.issubdtype(first.dtype, jax.dtypes.prng_key):
        data = np.stack([jax.random.key_data(value) for value in values])
        return jax.random.wrap_key_data(jnp.asarray(data))
    return jnp.asarray(np.stack(values))


def load_state_bank(path: str | Path) -> StateBank:
    """Load handoff entries and stack them into one device-resident bank.

    Legacy files containing only ``State`` remain readable. Their counters are
    initialized to zero, so they should only be used for policies that do not
    rely on episode-history features.
    """
    entries = []
    with open(path, "rb") as stream:
        while True:
            try:
                entries.append(pickle.load(stream))
            except EOFError:
                break
    if not entries:
        raise ValueError(f"state bank at {path} is empty")
    if isinstance(entries[0], StateBank):
        states = [entry.state for entry in entries]
        counters = [entry.counters for entry in entries]
        return StateBank(
            jax.tree.map(_stack_leaves, *states),
            jax.tree.map(_stack_leaves, *counters),
        )
    state = jax.tree.map(_stack_leaves, *entries)
    return StateBank(state, H.zeros(len(entries)))


def make_bank_start_fn(bank: StateBank) -> Callable[..., RolloutStart]:
    """Build a JIT-safe sampler returning matched state and history rows."""
    bank_size = jax.tree.leaves(bank.state)[0].shape[0]

    def bank_start(key, batch_size, board_size=None, starting_money=None):
        del board_size, starting_money
        indices = jax.random.randint(key, (batch_size,), 0, bank_size)
        state = jax.tree.map(lambda value: value[indices], bank.state)
        counters = jax.tree.map(lambda value: value[indices], bank.counters)
        assets = estimated_assets(state)
        return RolloutStart(state, counters, assets[:, 0] - assets[:, 1])

    return bank_start
