"""大量リプレイ対応のparallel JAX BC学習入口。"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import hydra
import jax
import optax
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from kaggriculture.policy.common.config import (
    ModelConfig,
    checkpoint_shape_metadata,
    validate_checkpoint_metadata,
)
from kaggriculture.policy.jax import model as M
from kaggriculture.policy.jax import policy as P
from kaggriculture.training.bc import core
from kaggriculture.training.bc.cache import iter_batches, prepare_episodes
from kaggriculture.training.checkpoint import (
    load_checkpoint,
    read_checkpoint_metadata,
    save_checkpoint,
)
from kaggriculture.training.replays import (
    list_episode_files,
    load_selected_sources,
    split_episode_files,
)
from kaggriculture.training.replays.state import CacheRules

logger = logging.getLogger(__name__)


def _restored_best(metadata: dict) -> float:
    """再開元checkpointのvalidation値をbest判定へ引き継ぐ。"""
    validation = metadata.get("best_validation_loss", metadata.get("validation_loss"))
    return float(validation) if validation is not None else float("inf")


def _sources(cfg):
    if cfg.data.selection_dir:
        directory = Path(to_absolute_path(cfg.data.selection_dir))
        train = load_selected_sources(directory, "train")
        validation = load_selected_sources(directory, "validation")
        if cfg.data.num_episodes is not None:
            rng = random.Random(cfg.data.split_seed)
            rng.shuffle(train)
            train = train[: cfg.data.num_episodes]
        return train, validation
    files = list_episode_files(Path(to_absolute_path(cfg.data.data_dir)))
    random.Random(cfg.data.split_seed).shuffle(files)
    if cfg.data.num_episodes is not None:
        files = files[: cfg.data.num_episodes]
    return split_episode_files(files, cfg.data.val_fraction, cfg.data.split_seed)


def _steps_per_epoch(paths, batch_size: int) -> int:
    """Return the number of complete training batches in one epoch."""
    total = 0
    for path in paths:
        stats = json.loads(path.with_suffix(".stats.json").read_text(encoding="utf-8"))
        total += stats["accepted"]
    return total // batch_size


def _schedule_steps(steps_per_epoch: int, max_epochs: int, max_steps: int) -> int:
    """Return the configured number of optimizer updates.

    Args:
        steps_per_epoch: Number of complete batches in one epoch.
        max_epochs: Maximum number of passes over the dataset.
        max_steps: Positive global update limit, or a non-positive value for no limit.

    Returns:
        Number of updates covered by the learning-rate schedule.

    Raises:
        ValueError: If the dataset cannot produce a complete batch.
    """
    if steps_per_epoch <= 0:
        raise ValueError("training data must contain at least one complete batch")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    epoch_steps = steps_per_epoch * max_epochs
    return min(epoch_steps, max_steps) if max_steps > 0 else epoch_steps


def _checkpoint(model, state, step, epoch, cfg, model_config, validation_paths, bc_config, best):
    """step_<n>を保存し、validationがこれまでの最良ならbestも更新する。"""
    validation = _validate(model, state.params, validation_paths, cfg, bc_config)
    improved = validation is not None and validation < best
    updated_best = validation if improved else best
    metadata = {
        **checkpoint_shape_metadata(),
        "trainer": "bc",
        "step": step,
        "epoch": epoch,
        "validation_loss": validation,
        "best_validation_loss": None if updated_best == float("inf") else updated_best,
        "model_config": asdict(model_config),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    directory = Path("checkpoints") / f"step_{step}"
    save_checkpoint(directory, state, metadata)
    if improved:
        save_checkpoint(Path("checkpoints/best"), state, metadata)
    return updated_best


def _validate(model, params, paths, cfg, bc_config):
    total = tokens = 0.0
    batches = iter_batches(paths, cfg.train.batch_size, seed=0, shuffle=False, drop_last=False)
    for batch in islice(batches, cfg.train.validation_batches):
        loss, count, _ = core.evaluate_minibatch(model, params, jax.device_put(batch), bc_config)
        loss, count = jax.device_get((loss, count))
        total += float(loss * count)
        tokens += float(count)
    return total / tokens if tokens else None


@hydra.main(version_base=None, config_path="../conf", config_name="bc")
def main(cfg: DictConfig) -> None:
    model_config = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    if model_config.use_asymmetric_critic:
        raise ValueError("BC trains actor only; use_asymmetric_critic must be false")
    train_sources, validation_sources = _sources(cfg)
    rules = CacheRules(
        turns_per_day=cfg.rules.turns_per_day,
        shed_capacity=cfg.rules.shed_capacity,
        hire_mult=cfg.rules.hire_mult,
        max_market_orders=cfg.rules.max_market_orders,
        min_player_reward=cfg.data.min_player_reward,
    )
    cache = Path(to_absolute_path(cfg.data.cache_dir))
    train_paths = prepare_episodes(train_sources, cache / "train", rules)
    validation_paths = prepare_episodes(validation_sources, cache / "validation", rules)
    model = M.PolicyValueNet(model_config)
    variables = P.initialize(model, jax.random.key(cfg.train.seed))
    steps_per_epoch = _steps_per_epoch(train_paths, cfg.train.batch_size)
    total_steps = _schedule_steps(steps_per_epoch, cfg.train.max_epochs, cfg.train.max_steps)
    learning_rate = optax.cosine_decay_schedule(
        cfg.train.learning_rate, total_steps, alpha=cfg.train.lr_decay_alpha
    )
    logger.info(
        "steps_per_epoch=%d total_steps=%d (max_epochs=%d)",
        steps_per_epoch,
        total_steps,
        cfg.train.max_epochs,
    )
    bc_config = core.BCConfig(
        learning_rate,
        cfg.train.weight_decay,
        cfg.train.max_grad_norm,
        cfg.rules.turns_per_day,
        cfg.rules.shed_capacity,
    )
    state = core.create_train_state(model, variables, bc_config)
    step = 0
    start_epoch = 0
    best = float("inf")
    if cfg.train.resume_checkpoint:
        directory = Path(to_absolute_path(cfg.train.resume_checkpoint))
        metadata = read_checkpoint_metadata(directory)
        validate_checkpoint_metadata(metadata)
        state, _ = load_checkpoint(directory, state)
        step = int(metadata["step"])
        start_epoch = int(metadata.get("epoch", 0))
        best = _restored_best(metadata)
    last_saved_step = step
    epoch = start_epoch
    for epoch in range(start_epoch, cfg.train.max_epochs):
        if 0 < cfg.train.max_steps <= step:
            break
        batches = iter_batches(
            train_paths,
            cfg.train.batch_size,
            seed=cfg.train.seed + epoch,
            shuffle=True,
            drop_last=True,
        )
        for batch in batches:
            if 0 < cfg.train.max_steps <= step:
                break
            state, (loss, _, excluded) = core.update_minibatch(
                model, state, jax.device_put(batch), bc_config
            )
            step += 1
            if step % cfg.train.log_interval == 0:
                loss, excluded = jax.device_get((loss, excluded))
                logger.info(
                    "epoch=%d step=%d loss=%.4f excluded=%d",
                    epoch,
                    step,
                    float(loss),
                    int(excluded),
                )
            if step % cfg.train.checkpoint_interval == 0:
                best = _checkpoint(
                    model, state, step, epoch, cfg, model_config, validation_paths, bc_config, best
                )
                last_saved_step = step
    if step > last_saved_step:
        _checkpoint(model, state, step, epoch, cfg, model_config, validation_paths, bc_config, best)


if __name__ == "__main__":
    main()
