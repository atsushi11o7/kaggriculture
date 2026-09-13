"""再現可能な学習用リプレイmanifestを作成する。"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from kaggriculture.training.replay_curation.analyze import analyze_episode
from kaggriculture.training.replays import list_episode_files, load_rating_manifest
from kaggriculture.training.replays.selection import SelectionEntry, write_entries

logger = logging.getLogger(__name__)


def _eligible(entry: SelectionEntry, cfg: DictConfig) -> bool:
    if cfg.selection.winner_only and entry.margin <= 0:
        return False
    if not cfg.selection.include_ties and entry.margin == 0:
        return False
    if entry.terminal_cash < cfg.selection.min_terminal_cash:
        return False
    if entry.margin < cfg.selection.min_margin:
        return False
    if cfg.selection.min_avg_agent_score is not None and (
        entry.avg_agent_score is None or entry.avg_agent_score < cfg.selection.min_avg_agent_score
    ):
        return False
    if cfg.selection.min_agent_score is not None and (
        entry.min_agent_score is None or entry.min_agent_score < cfg.selection.min_agent_score
    ):
        return False
    return True


def _limit(entries: list[SelectionEntry], cfg: DictConfig) -> list[SelectionEntry]:
    groups = defaultdict(list)
    for entry in entries:
        groups[entry.strategy].append(entry)
    selected = []
    for offset, strategy in enumerate(sorted(groups)):
        group = groups[strategy]
        random.Random(cfg.seed + offset).shuffle(group)
        limit = cfg.selection.max_per_strategy
        selected.extend(group if limit is None else group[:limit])
    random.Random(cfg.seed).shuffle(selected)
    limit = cfg.selection.max_total
    return selected if limit is None else selected[:limit]


def _assign_splits(entries: list[SelectionEntry], cfg: DictConfig) -> list[SelectionEntry]:
    """安定hashでepisode単位のsplitを作り、プレイヤー間の漏洩を防ぐ。"""
    result = []
    episode_splits = {}
    for entry in entries:
        if entry.episode_id not in episode_splits:
            digest = hashlib.sha256(f"{cfg.seed}:{entry.episode_id}".encode()).digest()
            fraction = int.from_bytes(digest[:8], "big") / 2**64
            if fraction < cfg.split.test_fraction:
                split = "test"
            elif fraction < cfg.split.test_fraction + cfg.split.validation_fraction:
                split = "validation"
            else:
                split = "train"
            episode_splits[entry.episode_id] = split
        result.append(replace(entry, split=episode_splits[entry.episode_id]))
    return sorted(result, key=lambda entry: (entry.split, entry.episode_id, entry.player))


def build_selection(cfg: DictConfig) -> list[SelectionEntry]:
    """設定に従って解析・選別・分割する。"""
    if cfg.split.validation_fraction < 0 or cfg.split.test_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if cfg.split.validation_fraction + cfg.split.test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1")
    replay_dir = Path(to_absolute_path(cfg.data.replay_dir))
    files = list_episode_files(replay_dir, cfg.data.num_episodes)
    ratings = {}
    if cfg.data.rating_manifest_dir is not None:
        ratings = load_rating_manifest(Path(to_absolute_path(cfg.data.rating_manifest_dir)))
    entries = []
    for index, path in enumerate(files, 1):
        try:
            entries.extend(analyze_episode(path, ratings.get(path.stem)))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            logger.warning("skipping %s: %s", path, error)
        if index % cfg.log_interval == 0:
            logger.info("analyzed %d/%d episodes", index, len(files))
    return _assign_splits(_limit([entry for entry in entries if _eligible(entry, cfg)], cfg), cfg)


@hydra.main(version_base=None, config_path="../conf", config_name="replay_curation")
def main(cfg: DictConfig) -> None:
    entries = build_selection(cfg)
    if not entries:
        raise ValueError("no replay players matched the selection criteria")
    output = Path(to_absolute_path(cfg.output_dir))
    for split in ("train", "validation", "test"):
        write_entries(output / f"{split}.jsonl", [e for e in entries if e.split == split])
    write_entries(output / "manifest.jsonl", entries)
    summary = {
        "num_entries": len(entries),
        "splits": Counter(entry.split for entry in entries),
        "strategies": Counter(entry.strategy for entry in entries),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("wrote %d selected player-replays to %s", len(entries), output)


if __name__ == "__main__":
    main()
