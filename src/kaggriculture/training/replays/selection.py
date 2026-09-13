"""選別済みプレイヤーmanifestの型・入出力・source検証。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SelectionEntry:
    """1試合の1プレイヤーを表す教師データ単位。"""

    episode_id: str
    path: str
    player: int
    split: str
    terminal_cash: float
    opponent_cash: float
    margin: float
    avg_agent_score: float | None
    min_agent_score: float | None
    create_time: str | None
    agent_name: str | None
    opponent_name: str | None
    strategy: str
    max_hands: int
    land_purchases: int
    hires: int
    sell_units: int
    buy_product_units: int
    source_size: int
    source_mtime_ns: int
    selection_reason: str = "winner"
    weight: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> SelectionEntry:
        return cls(**value)


def write_entries(path: Path, entries: list[SelectionEntry]) -> None:
    """JSONLをatomicに書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def load_split(directory: Path, split: str) -> list[SelectionEntry]:
    """選別ディレクトリのsplit JSONLを読み込む。"""
    path = directory / f"{split}.jsonl"
    with open(path, encoding="utf-8") as stream:
        return [SelectionEntry.from_dict(json.loads(line)) for line in stream if line.strip()]


def load_selected_sources(directory: Path, split: str) -> list[tuple[Path, tuple[int, ...]]]:
    """split manifestを読み、変更されていないsourceとplayerを返す。"""
    result = []
    for entry in load_split(directory, split):
        path = Path(entry.path)
        try:
            stat = path.stat()
        except OSError as error:
            raise ValueError(f"selected replay is unavailable: {path}") from error
        if stat.st_size != entry.source_size or stat.st_mtime_ns != entry.source_mtime_ns:
            raise ValueError(f"selected replay changed after curation: {path}")
        result.append((path, (entry.player,)))
    return result
