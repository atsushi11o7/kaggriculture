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
from kaggriculture.policy.jax import policy as P
from kaggriculture.policy.jax.model_factory import create_model, critic_version
from kaggriculture.training.bc import core
from kaggriculture.training.bc.cache import iter_batches, prepare_episodes
from kaggriculture.training.checkpoint import (
    load_checkpoint,
    read_checkpoint_metadata,
    save_checkpoint,
)
from kaggriculture.training.ppo.train import _load_actor_checkpoint, _load_value_checkpoint
from kaggriculture.training.replays import (
    list_episode_files,
    load_selected_sources,
    split_episode_files,
)
from kaggriculture.training.replays.state import CacheRules
from kaggriculture.training.rl import DailyRewardConfig

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


def _checkpoint(
    model,
    state,
    step,
    epoch,
    cfg,
    model_config,
    validation_paths,
    bc_config,
    best,
):
    metrics = _validate(model, state.params, validation_paths, cfg, bc_config)
    validation = float(metrics.loss) if metrics is not None else None
    improved = validation is not None and validation < best
    updated_best = validation if improved else best
    metadata = {
        **checkpoint_shape_metadata(),
        "trainer": "bc",
        "model_variant": cfg.train.model_variant,
        "step": step,
        "epoch": epoch,
        "validation_loss": validation,
        "validation_policy_loss": None if metrics is None else float(metrics.policy_loss),
        "validation_value_loss": None if metrics is None else float(metrics.value_loss),
        "best_validation_loss": None if updated_best == float("inf") else updated_best,
        "model_config": asdict(model_config),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if bc_config.value_loss_coefficient > 0:
        metadata.update(
            {
                "joint_value_training": True,
                "critic_architecture_version": critic_version(cfg.train.model_variant),
                "reward_mode": "terminal_win_daily_asset",
                "gamma": float(cfg.train.value_gamma),
                "daily_reward_coefficient": float(cfg.train.daily_reward_coefficient),
                "daily_reward_scale": float(cfg.train.daily_reward_scale),
                "daily_reward_maximum": float(cfg.train.daily_reward_maximum),
            }
        )
    directory = Path("checkpoints") / f"step_{step}"
    save_checkpoint(directory, state, metadata)
    if improved:
        save_checkpoint(Path("checkpoints/best"), state, metadata)
    return updated_best


def _load_bc_checkpoint_params(path: Path, variables: dict, model_config: ModelConfig) -> dict:
    return _load_actor_checkpoint(path, variables, model_config)


def _update_ema(ema, metrics, decay: float = 0.98):
    """ログ間隔ごとの1バッチ値は振れが大きいので、方策・value lossの移動平均も出す。"""
    values = (float(metrics.policy_loss), float(metrics.value_loss))
    if ema is None:
        return values
    return tuple(decay * old + (1 - decay) * new for old, new in zip(ema, values, strict=True))


def _spread_subset(paths, batch_size: int, batches: int):
    """検証に使うshardを、先頭ではなく全体から等間隔に選ぶ(1 shardは約700サンプル)。"""
    needed = max(1, -(-batches * batch_size // 700))
    stride = max(1, len(paths) // needed)
    return list(paths)[::stride][:needed]


def _validate(model, params, paths, cfg, bc_config):
    policy_total = policy_count = 0.0
    value_total = value_count = 0.0
    excluded = 0.0
    subset = _spread_subset(paths, cfg.train.batch_size, cfg.train.validation_batches)
    batches = iter_batches(subset, cfg.train.batch_size, seed=0, shuffle=False, drop_last=False)
    for batch in islice(batches, cfg.train.validation_batches):
        metrics = jax.device_get(
            core.evaluate_minibatch(model, params, jax.device_put(batch), bc_config)
        )
        weight = float(metrics.count)
        policy_total += float(metrics.policy_loss) * weight
        policy_count += weight
        excluded += float(metrics.excluded)
        samples = len(batch.players)
        value_total += float(metrics.value_loss) * samples
        value_count += samples
    if policy_count == 0:
        return None
    policy_loss = policy_total / policy_count
    value_loss = value_total / value_count
    loss = policy_loss + bc_config.value_loss_coefficient * value_loss
    return core.BCMetrics(loss, policy_loss, value_loss, policy_count, excluded)


@hydra.main(version_base=None, config_path="../conf", config_name="bc")
def main(cfg: DictConfig) -> None:
    model_config = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    joint_value_training = cfg.train.value_loss_coefficient > 0
    if model_config.use_asymmetric_critic != joint_value_training:
        raise ValueError(
            "use_asymmetric_critic must be true exactly when value_loss_coefficient is positive"
        )
    if cfg.train.value_gamma <= 0 or cfg.train.value_gamma > 1:
        raise ValueError("value_gamma must be in (0, 1]")
    DailyRewardConfig(
        cfg.train.daily_reward_coefficient,
        cfg.train.daily_reward_scale,
        cfg.train.daily_reward_maximum,
    )
    if (
        sum(
            bool(path)
            for path in (
                cfg.train.init_checkpoint,
                cfg.train.init_value_checkpoint,
                cfg.train.resume_checkpoint,
            )
        )
        > 1
    ):
        raise ValueError(
            "init_checkpoint, init_value_checkpoint and resume_checkpoint are mutually exclusive"
        )

    train_sources, validation_sources = _sources(cfg)
    rules = CacheRules(
        turns_per_day=cfg.rules.turns_per_day,
        shed_capacity=cfg.rules.shed_capacity,
        hire_mult=cfg.rules.hire_mult,
        max_market_orders=cfg.rules.max_market_orders,
        min_player_reward=cfg.data.min_player_reward,
    )
    cache = Path(to_absolute_path(cfg.data.cache_dir))
    reward = (
        {
            "gamma": cfg.train.value_gamma,
            "episode_steps": cfg.rules.episode_steps,
            "turns_per_day": cfg.rules.turns_per_day,
            "daily_reward_coefficient": cfg.train.daily_reward_coefficient,
            "daily_reward_scale": cfg.train.daily_reward_scale,
            "daily_reward_maximum": cfg.train.daily_reward_maximum,
        }
        if joint_value_training
        else None
    )
    train_paths = prepare_episodes(train_sources, cache / "train", rules, reward)
    validation_paths = prepare_episodes(validation_sources, cache / "validation", rules, reward)
    if joint_value_training and not (train_paths and validation_paths):
        raise ValueError("joint BC/value training requires complete train and validation games")

    model = create_model(model_config, cfg.train.model_variant)
    variables = P.initialize(model, jax.random.key(cfg.train.seed))
    if cfg.train.init_checkpoint:
        variables = _load_bc_checkpoint_params(
            Path(to_absolute_path(cfg.train.init_checkpoint)), variables, model_config
        )
    if cfg.train.init_value_checkpoint:
        variables = _load_value_checkpoint(
            Path(to_absolute_path(cfg.train.init_value_checkpoint)),
            variables,
            model_config,
            cfg.train.value_gamma,
            cfg.train.daily_reward_coefficient,
            cfg.train.daily_reward_scale,
            cfg.train.daily_reward_maximum,
        )

    steps_per_epoch = _steps_per_epoch(train_paths, cfg.train.batch_size)
    total_steps = _schedule_steps(steps_per_epoch, cfg.train.max_epochs, cfg.train.max_steps)
    learning_rate = optax.cosine_decay_schedule(
        cfg.train.learning_rate, total_steps, alpha=cfg.train.lr_decay_alpha
    )
    logger.info(
        "steps_per_epoch=%d total_steps=%d (max_epochs=%d) joint_value_training=%s",
        steps_per_epoch,
        total_steps,
        cfg.train.max_epochs,
        joint_value_training,
    )
    bc_config = core.BCConfig(
        learning_rate,
        cfg.train.weight_decay,
        cfg.train.max_grad_norm,
        cfg.rules.turns_per_day,
        cfg.rules.shed_capacity,
        cfg.train.value_loss_coefficient,
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
    ema = None
    for epoch in range(start_epoch, cfg.train.max_epochs):
        if 0 < cfg.train.max_steps <= step:
            break
        batches = iter_batches(
            train_paths,
            cfg.train.batch_size,
            seed=cfg.train.seed + epoch,
            shuffle=True,
            drop_last=True,
            mix_shards=cfg.train.shuffle_shards,
        )
        for batch in batches:
            if 0 < cfg.train.max_steps <= step:
                break
            state, metrics = core.update_minibatch(model, state, jax.device_put(batch), bc_config)
            step += 1
            if step % cfg.train.log_interval == 0:
                metrics = jax.device_get(metrics)
                ema = _update_ema(ema, metrics)
                logger.info(
                    "epoch=%d step=%d loss=%.4f policy_loss=%.4f value_loss=%.4f excluded=%d "
                    "policy_ema=%.4f value_ema=%.4f",
                    epoch,
                    step,
                    float(metrics.loss),
                    float(metrics.policy_loss),
                    float(metrics.value_loss),
                    int(metrics.excluded),
                    ema[0],
                    ema[1],
                )
            if step % cfg.train.checkpoint_interval == 0:
                best = _checkpoint(
                    model,
                    state,
                    step,
                    epoch,
                    cfg,
                    model_config,
                    validation_paths,
                    bc_config,
                    best,
                )
                last_saved_step = step
    if step > last_saved_step:
        _checkpoint(
            model,
            state,
            step,
            epoch,
            cfg,
            model_config,
            validation_paths,
            bc_config,
            best,
        )


if __name__ == "__main__":
    main()
