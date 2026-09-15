"""JAX学習状態の保存と復元。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import serialization
from flax.training.train_state import TrainState


def save_checkpoint(
    directory: Path, train_state: TrainState, metadata: Mapping[str, object]
) -> None:
    """TrainStateとJSON metadataをcheckpointディレクトリへ保存する。"""
    directory.mkdir(parents=True, exist_ok=True)
    state_tmp = directory / "state.msgpack.tmp"
    metadata_tmp = directory / "metadata.json.tmp"
    state_tmp.write_bytes(serialization.to_bytes(train_state))
    metadata_tmp.write_text(
        json.dumps(dict(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    state_tmp.replace(directory / "state.msgpack")
    metadata_tmp.replace(directory / "metadata.json")


def read_checkpoint_metadata(directory: Path) -> dict:
    """重いstate復元を行わずcheckpoint metadataだけを読む。"""
    return json.loads((directory / "metadata.json").read_text(encoding="utf-8"))


def load_checkpoint(directory: Path, target: TrainState) -> tuple[TrainState, dict]:
    """初期化済みtargetへcheckpointを復元する。"""
    restored = serialization.from_bytes(target, (directory / "state.msgpack").read_bytes())
    metadata = read_checkpoint_metadata(directory)
    # msgpack復元はNumPy配列を返す。そのままJIT内でindexすると
    # TracerArrayConversionErrorになるため、学習状態の配列leafをdeviceへ戻す。
    restored = restored.replace(
        step=jnp.asarray(restored.step),
        params=jax.tree.map(jnp.asarray, restored.params),
        opt_state=jax.tree.map(jnp.asarray, restored.opt_state),
    )
    return restored, metadata


def _is_prng_key(value) -> bool:
    return hasattr(value, "dtype") and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)


def _encode_keys(value):
    return jax.tree.map(
        lambda leaf: jax.random.key_data(leaf) if _is_prng_key(leaf) else leaf, value
    )


def save_pytree(path: Path, value) -> None:
    """JAX pytreeをatomicに保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(serialization.to_bytes(_encode_keys(value)))
    temporary.replace(path)


def load_pytree(path: Path, target):
    """保存済みpytreeをtargetの構造へ復元し、device配列へ変換する。"""
    restored = serialization.from_bytes(_encode_keys(target), path.read_bytes())
    return jax.tree.map(
        lambda value, template: (
            jax.random.wrap_key_data(value) if _is_prng_key(template) else jnp.asarray(value)
        ),
        restored,
        target,
    )
