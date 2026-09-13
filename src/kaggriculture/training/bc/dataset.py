"""BC(Behavior Cloning)用のリプレイデータセット。

`data/replays/`(gitignore対象、AGENTS.md参照)配下のエピソードJSONから
(観測, expert行動)の組をストリーミングで取り出す。生リプレイには
`distribution.validate_policy_action`がValueErrorにする無効な注文
(在庫0へのSELL、上限超過の数量等)が含まれるため、
`distribution.normalize_expert_action`で正規化してから最終確認する
(policy/README.mdの「BC用リプレイの入力契約」参照)。
"""

import csv
import gzip
import hashlib
import json
import logging
import os
import random
from collections.abc import Iterator
from pathlib import Path

import torch.utils.data

from kaggriculture.policy.torch import distribution as D
from kaggriculture.rules import constants as C

logger = logging.getLogger(__name__)
_CACHE_VERSION = 1


def list_episode_files(data_dir: Path, num_episodes: int | None = None) -> list[Path]:
    """data_dir以下(日付ディレクトリをまたいでもよい)のエピソードJSONを列挙する。

    Args:
        data_dir: エピソードJSONを再帰的に探すディレクトリ。
        num_episodes: 指定すれば、ソート後の先頭からこの件数だけに絞る
            (小規模なスモークテスト用)。

    Returns:
        ファイルパスの昇順ソート済みリスト(実行環境が変わっても順序が
        安定するようにするため)。
    """
    files = sorted(data_dir.glob("**/*.json"))
    if num_episodes is not None:
        files = files[:num_episodes]
    return files


def split_episode_files(
    files: list[Path], val_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """エピソード単位でtrain/valに分割する(ターン単位で分けるとリークするため)。

    Args:
        files: 分割対象のエピソードファイル一覧。
        val_fraction: valに回す割合([0, 1])。
        seed: シャッフルの再現用シード。

    Returns:
        (train用ファイル一覧, val用ファイル一覧)。
    """
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    if val_fraction > 0 and len(shuffled) > 1:
        n_val = max(1, n_val)
    return shuffled[n_val:], shuffled[:n_val]


def load_manifest(manifest_dir: Path) -> dict[str, dict]:
    """manifest_dir配下の全CSV(日付ごとのファイル)を読み込み、
    episode_id(str) -> 行(dict)のマップにまとめる。

    Args:
        manifest_dir: `episode_id,create_time,avg_score,min_score,sum_score,
            agent_count,size_bytes`の列を持つCSV群が置かれたディレクトリ
            (例: `data/replays/manifests/`)。

    Returns:
        episode_idをキーとする行の辞書。同じepisode_idが複数CSVに出ることは
        無い前提(日付ごとに1ファイルのため)。
    """
    manifest: dict[str, dict] = {}
    for csv_path in sorted(manifest_dir.glob("*.csv")):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                manifest[row["episode_id"]] = row
    return manifest


def filter_episodes_by_agent_score(
    files: list[Path],
    manifest: dict[str, dict],
    min_avg_agent_score: float | None = None,
    min_agent_score: float | None = None,
) -> list[Path]:
    """manifestのエージェントSkill Ratingでエピソードを絞り込む。

    強いエージェント同士の対局だけでfine-tuningしたい場合に使う。
    manifestに載っていないエピソード(ファイル名の拡張子抜きがepisode_idと一致
    しないもの)は、スコアが不明なため除外する。

    Args:
        files: 絞り込み対象のエピソードファイル一覧。
        manifest: `load_manifest`が返すepisode_id -> 行のマップ。
        min_avg_agent_score: 平均Skill Ratingの下限。
        min_agent_score: 低い方のエージェントのSkill Rating下限。

    Returns:
        条件を満たすファイルだけの一覧(元の順序を保つ)。
    """

    def _keep(path: Path) -> bool:
        row = manifest.get(path.stem)
        if row is None:
            return False
        if min_avg_agent_score is not None and float(row["avg_score"]) < min_avg_agent_score:
            return False
        if min_agent_score is not None and float(row["min_score"]) < min_agent_score:
            return False
        return True

    return [f for f in files if _keep(f)]


def iter_replay_samples(
    episode_path: Path, min_player_reward: float | None = None
) -> Iterator[tuple[dict, dict]]:
    """1エピソードJSONから、両プレイヤー分の(観測, 行動)の組を時系列に列挙する。

    kaggle-environmentsは、ターンiで受け取った観測を`steps[i]`に、
    それに対して返した行動を`steps[i + 1]`に記録する。そのため隣接stepを組にする。

    Args:
        episode_path: Kaggle episode JSONへのパス。
        min_player_reward: 指定時は、終端rewardがこの値以上のプレイヤーだけを列挙する。

    Yields:
        ターン開始時の観測と、その観測に対して提出された行動。
    """
    with open(episode_path) as f:
        data = json.load(f)
    steps = data["steps"]
    n_players = len(steps[0])
    rewards = data.get("rewards")
    if min_player_reward is not None and (
        not isinstance(rewards, list) or len(rewards) != n_players
    ):
        raise ValueError(f"missing player rewards in {episode_path}")
    eligible_players = [
        p
        for p in range(n_players)
        if min_player_reward is None or float(rewards[p]) >= min_player_reward
    ]
    for step_idx in range(1, len(steps)):
        for p in eligible_players:
            obs = steps[step_idx - 1][p]["observation"]
            action = steps[step_idx][p]["action"]
            if action is not None:
                yield obs, action


class ReplayActionDataset(torch.utils.data.IterableDataset):
    """エピソードファイル群から、正規化・検証済みの(観測, 行動)をyieldする。

    複数DataLoader workerがある場合、エピソード単位で分担する(1エピソード内の
    ターンを複数workerで分けたりはしない)。
    """

    def __init__(
        self,
        episode_files: list[Path],
        turns_per_day: int = 24,
        shed_capacity: int = 100,
        hire_mult: float = 1,
        max_market_orders: int = C.MAX_MARKET_ORDERS,
        min_player_reward: float | None = None,
        cache_dir: Path | None = None,
        validate_normalized: bool = False,
    ) -> None:
        self.episode_files = episode_files
        self.turns_per_day = turns_per_day
        self.shed_capacity = shed_capacity
        self.hire_mult = hire_mult
        self.max_market_orders = max_market_orders
        self.min_player_reward = min_player_reward
        self.cache_dir = cache_dir
        self.validate_normalized = validate_normalized

    def _kwargs(self) -> dict:
        return {
            "turns_per_day": self.turns_per_day,
            "shed_capacity": self.shed_capacity,
            "hire_mult": self.hire_mult,
            "max_market_orders": self.max_market_orders,
        }

    def _files_for_this_worker(self) -> list[Path]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.episode_files
        return self.episode_files[worker_info.id :: worker_info.num_workers]

    def _cache_path(self, episode_path: Path) -> Path | None:
        if self.cache_dir is None:
            return None
        stat = episode_path.stat()
        identity = (
            str(episode_path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            self.turns_per_day,
            self.shed_capacity,
            self.hire_mult,
            self.max_market_orders,
            self.min_player_reward,
            self.validate_normalized,
            _CACHE_VERSION,
        )
        digest = hashlib.sha256(repr(identity).encode()).hexdigest()
        return self.cache_dir / f"{digest}.jsonl.gz"

    def _normalized_samples(self, episode_path: Path) -> Iterator[tuple[dict, dict]]:
        kwargs = self._kwargs()
        for obs, action in iter_replay_samples(episode_path, self.min_player_reward):
            try:
                normalized = D.normalize_expert_action(obs, action, **kwargs)
                if self.validate_normalized:
                    D.validate_policy_action(obs, normalized, **kwargs)
            except ValueError as error:
                logger.warning("discarding unrepresentable sample from %s: %s", episode_path, error)
                continue
            yield obs, normalized

    def _episode_samples(self, episode_path: Path) -> Iterator[tuple[dict, dict]]:
        cache_path = self._cache_path(episode_path)
        if cache_path is not None and cache_path.exists():
            with gzip.open(cache_path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    obs, action = json.loads(line)
                    yield obs, action
            return
        if cache_path is None:
            yield from self._normalized_samples(episode_path)
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(f".tmp-{os.getpid()}")
        try:
            with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                for obs, action in self._normalized_samples(episode_path):
                    stream.write(json.dumps([obs, action], separators=(",", ":")) + "\n")
                    yield obs, action
            temporary.replace(cache_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def __iter__(self) -> Iterator[tuple[dict, dict]]:
        for episode_path in self._files_for_this_worker():
            try:
                yield from self._episode_samples(episode_path)
            except (OSError, json.JSONDecodeError, KeyError) as error:
                logger.warning("skipping unreadable episode %s: %s", episode_path, error)
