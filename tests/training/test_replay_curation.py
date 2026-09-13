"""再現可能なリプレイ選別のテスト。"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from kaggriculture.training.replay_curation.analyze import analyze_episode
from kaggriculture.training.replay_curation.select import _assign_splits, _eligible
from kaggriculture.training.replays.selection import (
    load_selected_sources,
    load_split,
    write_entries,
)

_CONFIG_DIR = Path(__file__).parents[2] / "src/kaggriculture/training/conf"


def _episode(path: Path) -> None:
    farms = [{"hands": []}, {"hands": [[1, 1], [2, 2]]}]
    path.write_text(
        json.dumps(
            {
                "id": 123,
                "rewards": [80_000, 40_000],
                "info": {"Agents": [{"Name": "winner"}, {"Name": "loser"}]},
                "steps": [
                    [
                        {
                            "observation": {"farms": farms},
                            "action": {"market": [["SELL", "MELON", 50], ["HIRE"]]},
                        },
                        {
                            "observation": {"farms": farms},
                            "action": {"market": [["SELL", "MILK", 10]]},
                        },
                    ]
                ],
            }
        )
    )


def test_analyze_and_select_winner(tmp_path: Path) -> None:
    path = tmp_path / "123.json"
    _episode(path)
    entries = analyze_episode(
        path, {"avg_score": "3000", "min_score": "2900", "create_time": "2026-09-01"}
    )
    cfg = OmegaConf.create(
        {
            "selection": {
                "winner_only": True,
                "include_ties": False,
                "min_terminal_cash": 50_000,
                "min_margin": 1,
                "min_avg_agent_score": 2500,
                "min_agent_score": 2500,
            }
        }
    )

    selected = [entry for entry in entries if _eligible(entry, cfg)]

    assert len(selected) == 1
    assert selected[0].player == 0
    assert selected[0].strategy == "crop_melon"
    assert selected[0].hires == 1
    assert entries[1].max_hands == 2


def test_split_is_deterministic_and_never_splits_episode(tmp_path: Path) -> None:
    path = tmp_path / "123.json"
    _episode(path)
    entries = analyze_episode(path)
    cfg = OmegaConf.create({"seed": 7, "split": {"validation_fraction": 0.2, "test_fraction": 0.2}})

    first = _assign_splits(entries, cfg)
    second = _assign_splits(entries, cfg)

    assert [entry.split for entry in first] == [entry.split for entry in second]
    assert len({entry.split for entry in first}) == 1


def test_manifest_round_trip_and_hydra_config(tmp_path: Path) -> None:
    path = tmp_path / "123.json"
    _episode(path)
    entry = replace(analyze_episode(path)[0], split="train")
    write_entries(tmp_path / "train.jsonl", [entry])

    assert load_split(tmp_path, "train") == [entry]
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR.resolve())):
        cfg = compose(config_name="replay_curation")
    assert cfg.selection.winner_only
    assert cfg.output_dir.endswith("broad_winners")


def test_selected_sources_validate_replay_identity(tmp_path: Path) -> None:
    path = tmp_path / "123.json"
    _episode(path)
    entry = replace(analyze_episode(path)[0], split="train")
    write_entries(tmp_path / "train.jsonl", [entry])

    assert load_selected_sources(tmp_path, "train") == [(path, (entry.player,))]

    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed after curation"):
        load_selected_sources(tmp_path, "train")
