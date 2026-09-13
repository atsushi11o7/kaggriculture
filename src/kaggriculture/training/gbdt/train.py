"""リプレイからGBDT教師方策を学習するHydra入口。"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from kaggriculture.training.gbdt.dataset import (
    BuildStats,
    RankingData,
    build_ranking_data,
    load_ranking_data,
    load_ranking_stats,
    save_ranking_data,
)
from kaggriculture.training.gbdt.features import FEATURE_VERSION
from kaggriculture.training.gbdt.model import GBDTRanker, RankerConfig
from kaggriculture.training.replays import (
    filter_episodes_by_agent_score,
    list_episode_files,
    load_rating_manifest,
    split_episode_files,
)

logger = logging.getLogger(__name__)


def _episode_files(cfg: DictConfig) -> tuple[list, list]:
    if cfg.data.selection_dir is not None:
        from kaggriculture.training.replays import load_selected_sources

        directory = Path(to_absolute_path(cfg.data.selection_dir))
        train = load_selected_sources(directory, "train")
        validation = load_selected_sources(directory, "validation")
        if not train:
            raise ValueError("selection manifest contains no training entries")
        return train, validation

    files = list_episode_files(Path(to_absolute_path(cfg.data.data_dir)))
    if cfg.data.manifest_dir is not None:
        manifest = load_rating_manifest(Path(to_absolute_path(cfg.data.manifest_dir)))
        files = filter_episodes_by_agent_score(
            files, manifest, cfg.data.min_avg_agent_score, cfg.data.min_agent_score
        )
    random.Random(cfg.data.seed).shuffle(files)
    if cfg.data.num_episodes is not None:
        files = files[: cfg.data.num_episodes]
    if not files:
        raise ValueError("no replay episodes found after filtering")
    return split_episode_files(files, cfg.data.val_fraction, cfg.data.seed)


def _cache_path(cfg: DictConfig, files: list) -> Path:
    sources = []
    for item in files:
        path, players = item if isinstance(item, tuple) else (item, None)
        sources.append((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns, players))
    identity = (
        FEATURE_VERSION,
        tuple(sources),
        cfg.data.min_player_reward,
        cfg.decode_rules.turns_per_day,
        cfg.decode_rules.shed_capacity,
        cfg.decode_rules.hire_mult,
        cfg.decode_rules.max_market_orders,
        cfg.decode_rules.episode_steps,
    )
    digest = hashlib.sha256(repr(identity).encode()).hexdigest()
    return Path(to_absolute_path(cfg.data.cache_dir)) / f"{digest}.npz"


def _load_or_build(cfg: DictConfig, files: list) -> tuple[dict[str, RankingData], BuildStats]:
    path = _cache_path(cfg, files)
    if path.exists() and not cfg.data.rebuild_cache:
        logger.info("loaded ranking cache %s", path)
        return load_ranking_data(path), load_ranking_stats(path)
    stats = BuildStats()
    data = build_ranking_data(
        files,
        turns_per_day=cfg.decode_rules.turns_per_day,
        shed_capacity=cfg.decode_rules.shed_capacity,
        hire_mult=cfg.decode_rules.hire_mult,
        max_market_orders=cfg.decode_rules.max_market_orders,
        min_player_reward=cfg.data.min_player_reward,
        episode_steps=cfg.decode_rules.episode_steps,
        stats=stats,
    )
    save_ranking_data(path, data, stats)
    return data, stats


@hydra.main(version_base=None, config_path="../conf", config_name="gbdt")
def main(cfg: DictConfig) -> None:
    train_files, validation_files = _episode_files(cfg)
    logger.info("ranking episodes: train=%d validation=%d", len(train_files), len(validation_files))
    datasets, train_stats = _load_or_build(cfg, train_files)
    if validation_files:
        validation, validation_stats = _load_or_build(cfg, validation_files)
    else:
        validation, validation_stats = None, BuildStats()
    Path("data_summary.json").write_text(
        json.dumps(
            {"train": train_stats.to_dict(), "validation": validation_stats.to_dict()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    config = RankerConfig(
        n_estimators=cfg.model.n_estimators,
        learning_rate=cfg.model.learning_rate,
        num_leaves=cfg.model.num_leaves,
        min_child_samples=cfg.model.min_child_samples,
        subsample=cfg.model.subsample,
        colsample_bytree=cfg.model.colsample_bytree,
        reg_lambda=cfg.model.reg_lambda,
        random_state=cfg.data.seed,
        n_jobs=cfg.model.n_jobs,
        early_stopping_rounds=cfg.model.early_stopping_rounds,
    )
    ranker = GBDTRanker.fit(datasets, config, validation)
    output = Path("checkpoint")
    ranker.save(output)
    for kind, data in datasets.items():
        train_accuracy = ranker.top1_accuracy(kind, data)
        validation_accuracy = (
            ranker.top1_accuracy(kind, validation[kind])
            if validation is not None and kind in validation
            else None
        )
        logger.info("%s top1: train=%.4f validation=%s", kind, train_accuracy, validation_accuracy)
    logger.info("saved GBDT teacher to %s", output.resolve())


if __name__ == "__main__":
    main()
