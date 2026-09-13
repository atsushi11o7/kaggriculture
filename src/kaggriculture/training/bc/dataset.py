"""PyTorch BC用のストリーミングリプレイデータセット。

`data/replays/`(gitignore対象、AGENTS.md参照)配下のエピソードJSONから
(観測, expert行動)の組をストリーミングで取り出す。生リプレイには
`distribution.validate_policy_action`がValueErrorにする無効な注文
(在庫0へのSELL、上限超過の数量等)が含まれるため、
`distribution.normalize_expert_action`で正規化してから最終確認する
(policy/README.mdの「BC用リプレイの入力契約」参照)。
"""

import gzip
import hashlib
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import torch.utils.data

from kaggriculture.policy.torch import distribution as D
from kaggriculture.rules import constants as C
from kaggriculture.training.replays.io import iter_replay_samples

logger = logging.getLogger(__name__)
_CACHE_VERSION = 1


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
