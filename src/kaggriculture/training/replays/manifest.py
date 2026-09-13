"""Kaggle episode Rating manifestの読込と絞り込み。"""

from __future__ import annotations

import csv
from pathlib import Path


def load_rating_manifest(manifest_dir: Path) -> dict[str, dict]:
    """ディレクトリ内のRating CSVをepisode IDで索引する。"""
    manifest = {}
    for path in sorted(manifest_dir.glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                manifest[row["episode_id"]] = row
    return manifest


def filter_episodes_by_agent_score(
    files: list[Path],
    manifest: dict[str, dict],
    min_avg_agent_score: float | None = None,
    min_agent_score: float | None = None,
) -> list[Path]:
    """対局時の平均・最低Skill Ratingでepisodeを絞り込む。"""

    def keep(path: Path) -> bool:
        row = manifest.get(path.stem)
        if row is None:
            return False
        if min_avg_agent_score is not None and float(row["avg_score"]) < min_avg_agent_score:
            return False
        return min_agent_score is None or float(row["min_score"]) >= min_agent_score

    return [path for path in files if keep(path)]
