"""共有Hydra規則と主要Python API既定値の契約テスト。"""

import inspect
from pathlib import Path

from hydra import compose, initialize_config_dir

from kaggriculture.policy.torch.candidate_api import decode_with_candidates
from kaggriculture.training.bc.jax_cache import CacheRules
from kaggriculture.training.ppo.rollout import RolloutConfig

_CONFIG_DIR = Path(__file__).parents[2] / "src/kaggriculture/training/conf"


def test_shared_rules_match_python_api_defaults() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR.resolve())):
        bc = compose(config_name="bc_jax")
        ppo = compose(config_name="ppo")
        gbdt = compose(config_name="gbdt")
        generate = compose(config_name="gbdt_generate", overrides=["model.checkpoint=/tmp/teacher"])

    rules = bc.rules
    assert ppo.rules == rules == gbdt.rules == generate.rules
    assert bc.data.turns_per_day == ppo.env.turns_per_day == rules.turns_per_day
    assert bc.data.shed_capacity == ppo.env.shed_capacity == rules.shed_capacity
    assert gbdt.decode_rules.episode_steps == ppo.env.episode_steps == rules.episode_steps
    assert generate.environment.maxMarketOrdersPerTurn == rules.max_market_orders

    cache = CacheRules()
    rollout = RolloutConfig()
    candidate_defaults = inspect.signature(decode_with_candidates).parameters
    assert cache.turns_per_day == rollout.turns_per_day == rules.turns_per_day
    assert cache.shed_capacity == rollout.shed_capacity == rules.shed_capacity
    assert rollout.episode_steps == rules.episode_steps
    assert candidate_defaults["turns_per_day"].default == rules.turns_per_day
    assert candidate_defaults["shed_capacity"].default == rules.shed_capacity
    assert candidate_defaults["max_market_orders"].default == rules.max_market_orders
