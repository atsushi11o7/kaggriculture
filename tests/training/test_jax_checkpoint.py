"""JAX checkpointの保存・復元テスト。"""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from kaggriculture.policy.jax import history as H
from kaggriculture.simulator.reset import reset
from kaggriculture.training.checkpoint import (
    load_checkpoint,
    load_pytree,
    save_checkpoint,
    save_pytree,
)


def test_restored_arrays_remain_usable_inside_jit(tmp_path) -> None:
    params = {"embedding": jnp.arange(12, dtype=jnp.float32).reshape(4, 3)}
    state = TrainState.create(
        apply_fn=lambda variables, index: variables["params"]["embedding"][index],
        params=params,
        tx=optax.adam(1e-3),
    )
    save_checkpoint(tmp_path, state, {"step": 0})

    restored, metadata = load_checkpoint(tmp_path, state)
    result = jax.jit(lambda p, index: p["embedding"][index])(restored.params, jnp.asarray([1, 3]))

    assert result.shape == (2, 3)
    assert metadata == {"step": 0}


def test_pytree_round_trip_preserves_runtime_arrays(tmp_path) -> None:
    target = {
        "key": jax.random.key(7),
        "state": reset(jax.random.key(8), 2),
        "counters": H.zeros(2),
        "nested": {"value": jnp.arange(6, dtype=jnp.float32).reshape(2, 3)},
    }
    path = tmp_path / "runtime.msgpack"

    save_pytree(path, target)
    restored = load_pytree(path, target)

    assert jnp.array_equal(restored["key"], target["key"])
    assert jnp.array_equal(restored["state"].money, target["state"].money)
    assert jnp.array_equal(restored["counters"].produced, target["counters"].produced)
    assert jnp.array_equal(restored["nested"]["value"], target["nested"]["value"])
