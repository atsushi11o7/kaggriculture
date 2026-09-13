"""固定shape replay cacheを使うJAX Behavior Cloning学習入口。"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import hydra
import jax
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import model as M
from kaggriculture.training.bc import jax_core
from kaggriculture.training.bc.jax_cache import CacheRules, iter_batches, prepare_episodes
from kaggriculture.training.checkpoint import load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


def _model_config(cfg: DictConfig) -> ModelConfig:
    values = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("model config must be a mapping")
    return ModelConfig(**values)


def _bc_config(cfg: DictConfig) -> jax_core.BCConfig:
    return jax_core.BCConfig(
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        turns_per_day=cfg.data.turns_per_day,
        shed_capacity=cfg.data.shed_capacity,
        hire_mult=cfg.data.hire_mult,
    )


def _validate_config(cfg: DictConfig) -> None:
    model = _model_config(cfg)
    if model.use_episode_history:
        raise ValueError("JAX BC replay counters are not implemented")
    if model.use_asymmetric_critic:
        raise ValueError("BC trains the actor only; use_asymmetric_critic must be false")
    if model.dropout != 0:
        raise ValueError("JAX BC currently requires model.dropout=0")
    if not 0 <= cfg.data.val_fraction < 1:
        raise ValueError("data.val_fraction must be in [0, 1)")
    if cfg.data.num_episodes is not None and cfg.data.num_episodes <= 0:
        raise ValueError("data.num_episodes must be positive or null")
    if cfg.data.cache_dir is None:
        raise ValueError("data.cache_dir is required for JAX BC")
    if cfg.train.batch_size <= 0 or cfg.train.max_epochs <= 0:
        raise ValueError("batch_size and max_epochs must be positive")
    if cfg.train.learning_rate <= 0 or cfg.train.max_grad_norm <= 0:
        raise ValueError("learning_rate and max_grad_norm must be positive")
    if cfg.train.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    for name in ("log_interval", "val_interval", "val_batches", "checkpoint_interval"):
        if cfg.train[name] <= 0:
            raise ValueError(f"train.{name} must be positive")
    if cfg.train.init_checkpoint is not None and cfg.train.init_weights is not None:
        raise ValueError("init_checkpoint and init_weights are mutually exclusive")


def _episode_files(cfg: DictConfig) -> tuple[list[Path], list[Path]]:
    # このimportはcache作成前のCPU処理だけで使い、学習step内には入らない。
    from kaggriculture.training.bc.dataset import (
        filter_episodes_by_agent_score,
        list_episode_files,
        load_manifest,
        split_episode_files,
    )

    files = list_episode_files(Path(to_absolute_path(cfg.data.data_dir)))
    if cfg.data.manifest_dir is not None:
        manifest = load_manifest(Path(to_absolute_path(cfg.data.manifest_dir)))
        files = filter_episodes_by_agent_score(
            files, manifest, cfg.data.min_avg_agent_score, cfg.data.min_agent_score
        )
    if cfg.data.num_episodes is not None:
        random.Random(cfg.data.split_seed).shuffle(files)
        files = files[: cfg.data.num_episodes]
    if not files:
        raise ValueError("no replay episodes found after filtering")
    train, validation = split_episode_files(files, cfg.data.val_fraction, cfg.data.split_seed)
    if not train or (cfg.data.val_fraction > 0 and not validation):
        raise ValueError("not enough episodes for the requested split")
    return train, validation


def _prepare_cache(cfg: DictConfig) -> tuple[list[Path], list[Path]]:
    train_sources, validation_sources = _episode_files(cfg)
    cache_dir = Path(to_absolute_path(cfg.data.cache_dir))
    rules = CacheRules(
        turns_per_day=cfg.data.turns_per_day,
        shed_capacity=cfg.data.shed_capacity,
        hire_mult=cfg.data.hire_mult,
        max_market_orders=cfg.data.max_market_orders,
        min_player_reward=cfg.data.min_player_reward,
    )
    train, train_created = prepare_episodes(train_sources, cache_dir, rules)
    validation, validation_created = prepare_episodes(validation_sources, cache_dir, rules)
    logger.info(
        "cache shards: train=%d (%d new), validation=%d (%d new)",
        len(train),
        train_created,
        len(validation),
        validation_created,
    )
    if not train:
        raise ValueError("no representable training samples were cached")
    return train, validation


def _prune_checkpoints(directory: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    paths = sorted(directory.glob("step_*"), key=lambda path: int(path.name.removeprefix("step_")))
    for path in paths[:-keep_last]:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()


def _save(
    directory: Path,
    train_state,
    step: int,
    epoch: int,
    batch_index: int,
    model_config: ModelConfig,
    bc_config: jax_core.BCConfig,
    resolved_config: dict,
) -> None:
    save_checkpoint(
        directory,
        train_state,
        {
            "trainer": "jax_bc",
            "step": step,
            "epoch": epoch,
            "batch_index": batch_index,
            "model_config": asdict(model_config),
            "bc_config": asdict(bc_config),
            "config": resolved_config,
        },
    )


def _load_initial_weights(path: Path, train_state, model, config: jax_core.BCConfig):
    """JAX BC checkpointからoptimizerを除く重みだけを読み込む。"""
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    source_config = ModelConfig(**metadata["model_config"])
    source_model = M.PolicyValueNet(source_config)
    source_variables = M.initialize(source_model, jax.random.key(0))
    source_bc = jax_core.BCConfig(**metadata["bc_config"])
    source_state = jax_core.create_train_state(source_model, source_variables, source_bc)
    restored, _ = load_checkpoint(path, source_state)
    if jax.tree.structure(restored.params) != jax.tree.structure(train_state.params):
        raise ValueError("initial checkpoint model structure does not match")
    return train_state.replace(params=restored.params)


def _validate(model, params, paths: list[Path], cfg: DictConfig, bc_config) -> float | None:
    total_loss = 0.0
    total_tokens = 0.0
    for index, host_batch in enumerate(
        iter_batches(
            paths,
            cfg.train.batch_size,
            seed=cfg.train.seed,
            shuffle=False,
            drop_last=False,
        )
    ):
        if index >= cfg.train.val_batches:
            break
        metrics = jax_core.evaluate_minibatch(model, params, jax.device_put(host_batch), bc_config)
        loss, tokens = jax.device_get((metrics.loss, metrics.num_tokens))
        total_loss += float(loss * tokens)
        total_tokens += float(tokens)
    return total_loss / total_tokens if total_tokens else None


@hydra.main(version_base=None, config_path="../conf", config_name="bc_jax")
def main(cfg: DictConfig) -> None:
    _validate_config(cfg)
    train_paths, validation_paths = _prepare_cache(cfg)
    model_config = _model_config(cfg)
    bc_config = _bc_config(cfg)
    model = M.PolicyValueNet(model_config)
    key = jax.random.key(cfg.train.seed)
    key, init_key = jax.random.split(key)
    variables = M.initialize(model, init_key)
    train_state = jax_core.create_train_state(model, variables, bc_config)
    step = 0
    start_epoch = 0
    start_batch_index = 0

    if cfg.train.init_checkpoint is not None:
        path = Path(to_absolute_path(cfg.train.init_checkpoint))
        train_state, metadata = load_checkpoint(path, train_state)
        step = int(metadata["step"])
        start_epoch = int(metadata["epoch"])
        start_batch_index = int(metadata.get("batch_index", 0))
        logger.info("resumed checkpoint %s at step %d", path, step)
    elif cfg.train.init_weights is not None:
        path = Path(to_absolute_path(cfg.train.init_weights))
        train_state = _load_initial_weights(path, train_state, model, bc_config)
        logger.info("loaded initial weights %s", path)

    checkpoint_root = Path("checkpoints")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    metrics_path = Path("metrics.jsonl")
    best_validation = float("inf")
    cursor_epoch = start_epoch
    cursor_batch_index = start_batch_index

    for epoch in range(start_epoch, cfg.train.max_epochs):
        batches = iter_batches(
            train_paths,
            cfg.train.batch_size,
            seed=cfg.train.seed + epoch,
            shuffle=True,
            drop_last=True,
        )
        batch_offset = start_batch_index if epoch == start_epoch else 0
        batches = islice(batches, batch_offset, None)
        for batch_index, host_batch in enumerate(batches, start=batch_offset + 1):
            if cfg.train.max_steps > 0 and step >= cfg.train.max_steps:
                break
            train_state, metrics = jax_core.update_minibatch(
                model, train_state, jax.device_put(host_batch), bc_config
            )
            step += 1
            cursor_epoch = epoch
            cursor_batch_index = batch_index
            if step % cfg.train.log_interval == 0:
                values = jax.device_get(metrics)
                logger.info(
                    "epoch=%d step=%d loss=%.4f entropy=%.4f",
                    epoch,
                    step,
                    values.loss,
                    values.entropy,
                )
                with open(metrics_path, "a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch,
                                "train_loss": float(values.loss),
                                "train_entropy": float(values.entropy),
                            }
                        )
                        + "\n"
                    )
            if validation_paths and step % cfg.train.val_interval == 0:
                validation = _validate(model, train_state.params, validation_paths, cfg, bc_config)
                if validation is not None:
                    logger.info("step=%d validation_loss=%.4f", step, validation)
                    if validation < best_validation:
                        best_validation = validation
                        _save(
                            checkpoint_root / "best",
                            train_state,
                            step,
                            epoch,
                            batch_index,
                            model_config,
                            bc_config,
                            resolved,
                        )
            if step % cfg.train.checkpoint_interval == 0:
                _save(
                    checkpoint_root / f"step_{step}",
                    train_state,
                    step,
                    epoch,
                    batch_index,
                    model_config,
                    bc_config,
                    resolved,
                )
                _prune_checkpoints(checkpoint_root, cfg.train.keep_last_checkpoints)
        if cfg.train.max_steps > 0 and step >= cfg.train.max_steps:
            break
        cursor_epoch = epoch + 1
        cursor_batch_index = 0

    if step == 0:
        raise ValueError("cache does not contain one full training batch")
    if validation_paths:
        validation = _validate(model, train_state.params, validation_paths, cfg, bc_config)
        if validation is not None and validation < best_validation:
            _save(
                checkpoint_root / "best",
                train_state,
                step,
                cursor_epoch,
                cursor_batch_index,
                model_config,
                bc_config,
                resolved,
            )
    if step % cfg.train.checkpoint_interval:
        _save(
            checkpoint_root / f"step_{step}",
            train_state,
            step,
            cursor_epoch,
            cursor_batch_index,
            model_config,
            bc_config,
            resolved,
        )


if __name__ == "__main__":
    main()
