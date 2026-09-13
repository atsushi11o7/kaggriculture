"""JAX checkpointの保存・復元テスト。"""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from kaggriculture.training.checkpoint import load_checkpoint, save_checkpoint


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
