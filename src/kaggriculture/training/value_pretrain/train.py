"""固定BC Actorと同じモデルの非対称criticを完了試合で事前学習する。"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict
from pathlib import Path

import hydra
import jax
import optax
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from kaggriculture.policy.common.config import (
    ModelConfig,
    checkpoint_shape_metadata,
)
from kaggriculture.policy.jax import policy as P
from kaggriculture.policy.jax.model_factory import create_model, critic_version
from kaggriculture.training.checkpoint import save_checkpoint
from kaggriculture.training.ppo.checkpointing import load_actor_checkpoint
from kaggriculture.training.replays import (
    list_episode_files,
    load_selected_sources,
    split_episode_files,
)
from kaggriculture.training.rl import DailyRewardConfig
from kaggriculture.training.value_pretrain import cache, core
from kaggriculture.training.value_pretrain.evaluation import (
    compute_metrics,
    sample_fixed_states,
    train_target_mean,
)

logger = logging.getLogger(__name__)


def _sources(cfg):
    if cfg.data.selection_dir:
        directory = Path(to_absolute_path(cfg.data.selection_dir))
        train = list(dict.fromkeys(path for path, _ in load_selected_sources(directory, "train")))
        validation = list(
            dict.fromkeys(path for path, _ in load_selected_sources(directory, "validation"))
        )
        if cfg.data.num_episodes is not None:
            train = train[: cfg.data.num_episodes]
        if set(train) & set(validation):
            raise ValueError("train and validation manifests share an episode")
        return train, validation
    files = list_episode_files(Path(to_absolute_path(cfg.data.data_dir)))
    random.Random(cfg.data.split_seed).shuffle(files)
    if cfg.data.num_episodes is not None:
        files = files[: cfg.data.num_episodes]
    return split_episode_files(files, cfg.data.val_fraction, cfg.data.split_seed)


def _steps_per_epoch(num_episodes: int, episode_steps: int, batch_size: int) -> int:
    """完了試合数から1epochあたりのおおよそのbatch数を見積もる。

    不完全なエピソードは`load_episode`が読み込み時に捨てるため、この見積もりは
    実際よりわずかに多く出ることがある(scheduleの想定step数としては問題ない)。
    """
    return (num_episodes * (episode_steps - 1)) // batch_size


def _schedule_steps(steps_per_epoch: int, max_epochs: int, max_steps: int) -> int:
    """PPO/BCと同じ規約でschedule全体のstep数を返す。"""
    if steps_per_epoch <= 0:
        raise ValueError("training data must contain at least one complete batch")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    epoch_steps = steps_per_epoch * max_epochs
    return min(epoch_steps, max_steps) if max_steps > 0 else epoch_steps


def _reward_kwargs(cfg):
    return {
        "gamma": cfg.train.gamma,
        "episode_steps": cfg.rules.episode_steps,
        "turns_per_day": cfg.rules.turns_per_day,
        "daily_reward_coefficient": cfg.train.daily_reward_coefficient,
        "daily_reward_scale": cfg.train.daily_reward_scale,
        "daily_reward_maximum": cfg.train.daily_reward_maximum,
    }


@hydra.main(version_base=None, config_path="../conf", config_name="value_pretrain")
def main(cfg: DictConfig) -> None:
    if cfg.train.actor_checkpoint is None:
        raise ValueError("train.actor_checkpoint is required")
    if not 0 < cfg.train.gamma <= 1:
        raise ValueError("train.gamma must be in (0, 1]")
    DailyRewardConfig(
        cfg.train.daily_reward_coefficient,
        cfg.train.daily_reward_scale,
        cfg.train.daily_reward_maximum,
    )
    model_config = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    train_paths, validation_paths = _sources(cfg)
    if not train_paths or not validation_paths:
        raise ValueError("training and validation both require completed episodes")
    reward_kwargs = _reward_kwargs(cfg)
    cache_dir = Path(to_absolute_path(cfg.data.cache_dir))
    train_shards = cache.prepare_episodes(train_paths, cache_dir / "train", **reward_kwargs)
    validation_shards = cache.prepare_episodes(
        validation_paths, cache_dir / "validation", **reward_kwargs
    )
    if not train_shards or not validation_shards:
        raise ValueError("training and validation both require completed episodes")
    model = create_model(model_config)
    variables = P.initialize(model, jax.random.key(cfg.train.seed))
    variables = load_actor_checkpoint(
        Path(to_absolute_path(cfg.train.actor_checkpoint)), variables, model_config
    )
    steps_per_epoch = _steps_per_epoch(
        len(train_shards), cfg.rules.episode_steps, cfg.train.batch_size
    )
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
    state = core.create_train_state(
        model,
        variables,
        learning_rate=learning_rate,
        max_grad_norm=cfg.train.max_grad_norm,
    )
    update = jax.jit(core.update, static_argnums=(0, 4))
    best = float("inf")
    steps = 0
    run_dir = Path.cwd()

    # 検証は試合単位で固定サンプリングし、epochを跨いで同じ局面集合を使う
    # (毎回先頭バッチだけを見る旧実装は、少数試合への偏りが大きかった)。
    train_mean = train_target_mean(train_shards)
    evaluation_set = sample_fixed_states(
        validation_shards,
        episodes=cfg.validation.episodes,
        states_per_episode=cfg.validation.states_per_episode,
        seed=cfg.validation.seed,
        turns_per_day=cfg.rules.turns_per_day,
    )
    logger.info(
        "validation set: %d episodes x %d states = %d rows",
        len(validation_shards) if cfg.validation.episodes is None else cfg.validation.episodes,
        cfg.validation.states_per_episode,
        evaluation_set.targets.shape[0],
    )
    for epoch in range(cfg.train.max_epochs):
        batches = cache.iter_batches(
            train_shards,
            batch_size=cfg.train.batch_size,
            seed=cfg.train.seed + epoch,
            shuffle=True,
            drop_last=True,
        )
        for states, targets in batches:
            state, loss = update(model, state, states, targets, cfg.rules.turns_per_day)
            steps += 1
            if steps % cfg.train.log_interval == 0:
                logger.info("step=%d epoch=%d train_loss=%.6f", steps, epoch + 1, float(loss))
            if 0 < cfg.train.max_steps <= steps:
                break
        if steps == 0:
            raise ValueError("training set has no complete batches")
        metrics = compute_metrics(
            model,
            state.params,
            evaluation_set,
            train_mean=train_mean,
            turns_per_day=cfg.rules.turns_per_day,
        )
        validation = metrics["model_mse"]
        metadata = {
            "trainer": "value_pretrain",
            "critic_architecture_version": critic_version(),
            "model_config": asdict(model_config),
            **checkpoint_shape_metadata(),
            "reward_mode": "terminal_win_daily_asset",
            "gamma": float(cfg.train.gamma),
            "daily_reward_coefficient": float(cfg.train.daily_reward_coefficient),
            "daily_reward_scale": float(cfg.train.daily_reward_scale),
            "daily_reward_maximum": float(cfg.train.daily_reward_maximum),
            "step": steps,
            "epoch": epoch + 1,
            "validation_loss": validation,
            "validation_metrics": metrics,
            "actor_checkpoint": str(Path(to_absolute_path(cfg.train.actor_checkpoint))),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        save_checkpoint(run_dir / "checkpoints" / "last", state, metadata)
        if validation < best:
            best = validation
            save_checkpoint(run_dir / "checkpoints" / "best", state, metadata)
        logger.info(
            "step=%d epoch=%d validation_loss=%.6f best=%.6f r2_vs_mean_baseline=%.4f "
            "sign_last5days=%.3f",
            steps,
            epoch + 1,
            validation,
            best,
            metrics["r2_vs_mean_baseline"],
            metrics["sign_accuracy_last5days"],
        )
        if 0 < cfg.train.max_steps <= steps:
            break


if __name__ == "__main__":
    main()
