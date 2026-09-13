"""合成observation(実データ非依存)を使うテスト用フィクスチャ。

data/はgitignore対象(AGENTS.md参照)のため、リポジトリには実際のリプレイデータが
無く、CIや他の開発者の環境にも存在しない。ここでは初期状態(day 0、両者ともNW
区画のみ解放・所持金3000・納屋空)のobservationを直接組み立て、実データ無しで
act()/evaluate_actions()系のテストが完結するようにする。
"""

from dataclasses import replace

import pytest
import torch

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.torch import model as M
from kaggriculture.simulator import constants as C

BOARD_SIZE = 10


DEFAULT_MODEL_CONFIG = ModelConfig(
    d_model=128,
    num_heads=4,
    d_feedforward=512,
    num_layers_encoder=4,
    num_layers_decoder=3,
    dropout=0.0,
    use_episode_history=False,
    use_asymmetric_critic=False,
    num_layers_critic=2,
)


def test_model_config(**changes) -> ModelConfig:
    """テスト用の標準モデル構成を一部だけ上書きする。"""
    return replace(DEFAULT_MODEL_CONFIG, **changes)


def _fresh_farm() -> dict:
    half = BOARD_SIZE // 2
    tiles = [
        [None if y < half and x < half else "LOCKED" for x in range(BOARD_SIZE)]
        for y in range(BOARD_SIZE)
    ]
    return {
        "farmer": [half - 1, half - 1],  # 納屋隣接マス(board.default_spawn_position参照)
        "hands": [],
        "hires_today": 0,
        "money": 3000.0,
        "tiles": tiles,
        "unlocked_quadrants": ["NW"],
    }


def make_fresh_observation(player: int = 0, day: int = 0) -> dict:
    """kaggriculture初期状態(day 0、reset.py相当)のobservationを1人称分組み立てる。"""
    return {
        "day": day,
        "hour": 0,
        "step": 0,
        "player": player,
        "farms": [_fresh_farm(), _fresh_farm()],
        "private": {
            "inventories": [{}],
            "seeds": dict.fromkeys(C.CROPS, 0),
            "shed": dict.fromkeys(C.SHED_ITEMS, 0),
        },
        "market": {
            "inventory": dict.fromkeys(C.PRODUCTS, 10_000),
            "prices": dict.fromkeys(C.PRODUCTS, 100),
        },
        "town": {"unlocked_shops": []},
        "remainingOverageTime": 60,
    }


def make_fresh_private() -> dict:
    """obs["private"]と同じ形の、空の(=day 0相当の)非公開情報。opponent_privateの
    テスト用に使う。"""
    return {
        "inventories": [{}],
        "seeds": dict.fromkeys(C.CROPS, 0),
        "shed": dict.fromkeys(C.SHED_ITEMS, 0),
    }


@pytest.fixture
def fresh_private() -> dict:
    return make_fresh_private()


@pytest.fixture
def net() -> M.PolicyValueNet:
    torch.manual_seed(0)
    return M.PolicyValueNet(DEFAULT_MODEL_CONFIG)


@pytest.fixture
def net_with_history() -> M.PolicyValueNet:
    torch.manual_seed(0)
    return M.PolicyValueNet(test_model_config(use_episode_history=True))


@pytest.fixture
def net_with_asymmetric_critic() -> M.PolicyValueNet:
    torch.manual_seed(0)
    return M.PolicyValueNet(test_model_config(use_asymmetric_critic=True))


@pytest.fixture
def fresh_obs() -> dict:
    return make_fresh_observation()
