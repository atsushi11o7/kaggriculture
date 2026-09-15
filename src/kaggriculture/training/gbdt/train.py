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
    RankingFiles,
    build_ranking_files,
    load_ranking_files,
    open_ranking_data,
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


def _validate_config(cfg: DictConfig) -> None:
    if cfg.data.quantity_candidates < 3:
        raise ValueError("data.quantity_candidates must be at least 3")
    if cfg.data.num_episodes is not None and cfg.data.num_episodes <= 0:
        raise ValueError("data.num_episodes must be positive or null")
    if not 0 <= cfg.data.val_fraction < 1:
        raise ValueError("data.val_fraction must be in [0, 1)")


def _episode_files(cfg: DictConfig) -> tuple[list, list]:
    if cfg.data.selection_dir is not None:
        from kaggriculture.training.replays import load_selected_sources

        directory = Path(to_absolute_path(cfg.data.selection_dir))
        train = load_selected_sources(directory, "train")
        validation = load_selected_sources(directory, "validation")
        if not train:
            raise ValueError("selection manifest contains no training entries")
        if cfg.data.num_episodes is not None:
            # スモークテスト用の件数制限。順序偏りを避けてから切り、
            # validationも元の比率に合わせて縮小する。
            val_ratio = len(validation) / max(len(train), 1)
            random.Random(cfg.data.seed).shuffle(train)
            random.Random(cfg.data.seed).shuffle(validation)
            train = train[: cfg.data.num_episodes]
            validation = validation[: max(1, round(cfg.data.num_episodes * val_ratio))]
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
    return Path(to_absolute_path(cfg.data.cache_dir)) / f"{digest}-q{cfg.data.quantity_candidates}"


def _load_or_build(cfg: DictConfig, files: list) -> RankingFiles:
    path = _cache_path(cfg, files)
    if (path / "metadata.json").exists() and not cfg.data.rebuild_cache:
        logger.info("loaded ranking cache %s", path)
        return load_ranking_files(path)
    data = build_ranking_files(
        files,
        path,
        turns_per_day=cfg.decode_rules.turns_per_day,
        shed_capacity=cfg.decode_rules.shed_capacity,
        hire_mult=cfg.decode_rules.hire_mult,
        max_market_orders=cfg.decode_rules.max_market_orders,
        min_player_reward=cfg.data.min_player_reward,
        episode_steps=cfg.decode_rules.episode_steps,
        quantity_limit=cfg.data.quantity_candidates,
    )
    return data


@hydra.main(version_base=None, config_path="../conf", config_name="gbdt")
def main(cfg: DictConfig) -> None:
    _validate_config(cfg)
    train_files, validation_files = _episode_files(cfg)
    logger.info("ranking episodes: train=%d validation=%d", len(train_files), len(validation_files))
    datasets = _load_or_build(cfg, train_files)
    validation = _load_or_build(cfg, validation_files) if validation_files else None
    train_stats = datasets.stats
    validation_stats = validation.stats if validation is not None else BuildStats()
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
    class_reports = {}
    for kind, info in datasets.kinds.items():
        if not info["rows"]:
            continue
        data = open_ranking_data(datasets, kind)
        train_accuracy = ranker.top1_accuracy(kind, data)
        validation_data = (
            open_ranking_data(validation, kind)
            if validation is not None and validation.kinds[kind]["rows"]
            else None
        )
        validation_accuracy = (
            ranker.top1_accuracy(kind, validation_data) if validation_data is not None else None
        )
        logger.info("%s top1: train=%.4f validation=%s", kind, train_accuracy, validation_accuracy)
        # market_opは全体top1だけだとSTOP/HIREの高精度に隠れて、BUY_SEED等の
        # 少数派だが決定的に重要なクラスの性能が見えない。クラス別・macro精度・
        # 予測分布を別途記録する。
        if kind == "market_op":
            train_report = ranker.class_top1_report(kind, data)
            validation_report = (
                ranker.class_top1_report(kind, validation_data)
                if validation_data is not None
                else None
            )
            logger.info(
                "%s macro top1 (exact/op): train=%.4f/%.4f validation=%s/%s",
                kind,
                train_report["macro_accuracy"],
                train_report["macro_op_accuracy"],
                f"{validation_report['macro_accuracy']:.4f}" if validation_report else None,
                f"{validation_report['macro_op_accuracy']:.4f}" if validation_report else None,
            )
            # exact: 候補の完全一致(作物まで一致必須)。op: op名だけの一致
            # (「その種類の行動を選ぶ判断自体ができたか」。BUY_SEEDの別作物を
            # 選んでも正解扱い)。両方見ないと、opは正しく選べているのに作物の
            # 好みが実データと違うだけのケースを、多数派崩壊と誤認しかねない。
            for cls in sorted(train_report["per_class_count"]):
                validation_exact = (
                    validation_report["per_class_accuracy"].get(cls) if validation_report else None
                )
                validation_op = (
                    validation_report["per_class_op_accuracy"].get(cls)
                    if validation_report
                    else None
                )
                logger.info(
                    "  %s: n=%d train_acc(exact/op)=%.3f/%.3f validation_acc(exact/op)=%s/%s",
                    cls,
                    train_report["per_class_count"][cls],
                    train_report["per_class_accuracy"][cls],
                    train_report["per_class_op_accuracy"][cls],
                    f"{validation_exact:.3f}" if validation_exact is not None else None,
                    f"{validation_op:.3f}" if validation_op is not None else None,
                )
            class_reports[kind] = {"train": train_report, "validation": validation_report}
    if class_reports:
        Path("class_report.json").write_text(
            json.dumps(class_reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    logger.info("saved GBDT teacher to %s", output.resolve())


if __name__ == "__main__":
    main()
