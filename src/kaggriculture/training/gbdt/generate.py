"""GBDT教師同士の対局を既存BCが読めるリプレイへ保存する。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from kaggle_environments import make
from omegaconf import DictConfig, OmegaConf

from kaggriculture.training.gbdt.agent import GBDTAgent
from kaggriculture.training.gbdt.model import GBDTRanker

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="gbdt_generate")
def main(cfg: DictConfig) -> None:
    ranker = GBDTRanker.load(Path(to_absolute_path(cfg.model.checkpoint)))
    output_dir = Path(to_absolute_path(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    environment_config = OmegaConf.to_container(cfg.environment, resolve=True)
    if not isinstance(environment_config, dict):
        raise TypeError("environment config must be a mapping")
    if cfg.num_games <= 0 or cfg.log_interval <= 0:
        raise ValueError("num_games and log_interval must be positive")

    for game in range(cfg.num_games):
        path = output_dir / f"gbdt-{cfg.seed + game}.json"
        if path.exists() and not cfg.overwrite:
            raise FileExistsError(f"replay already exists: {path}")
        game_config = dict(environment_config)
        game_config["seed"] = cfg.seed + game
        agent0 = GBDTAgent(
            ranker,
            turns_per_day=game_config["turnsPerDay"],
            shed_capacity=game_config["shedCapacity"],
            hire_mult=game_config["farmHandCostMult"],
            max_market_orders=game_config["maxMarketOrdersPerTurn"],
            episode_steps=game_config["episodeSteps"],
            temperature=cfg.temperature,
            seed=cfg.seed + 2 * game,
        )
        agent1 = GBDTAgent(
            ranker,
            turns_per_day=game_config["turnsPerDay"],
            shed_capacity=game_config["shedCapacity"],
            hire_mult=game_config["farmHandCostMult"],
            max_market_orders=game_config["maxMarketOrdersPerTurn"],
            episode_steps=game_config["episodeSteps"],
            temperature=cfg.temperature,
            seed=cfg.seed + 2 * game + 1,
        )
        env = make("kaggriculture", configuration=game_config, debug=cfg.debug)
        env.run([agent0, agent1])
        path.write_text(json.dumps(env.toJSON(), ensure_ascii=False), encoding="utf-8")
        if (game + 1) % cfg.log_interval == 0:
            logger.info("generated %d/%d games", game + 1, cfg.num_games)


if __name__ == "__main__":
    main()
