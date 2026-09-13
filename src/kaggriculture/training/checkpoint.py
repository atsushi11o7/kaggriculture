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


def load_checkpoint(directory: Path, target: TrainState) -> tuple[TrainState, dict]:
    """初期化済みtargetへcheckpointを復元する。"""
    restored = serialization.from_bytes(target, (directory / "state.msgpack").read_bytes())
    # msgpack復元はNumPy配列を返す。そのままJIT内でindexすると
    # TracerArrayConversionErrorになるため、学習状態の配列leafをdeviceへ戻す。
    restored = restored.replace(
        step=jnp.asarray(restored.step),
        params=jax.tree.map(jnp.asarray, restored.params),
        opt_state=jax.tree.map(jnp.asarray, restored.opt_state),
    )
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return restored, metadata
