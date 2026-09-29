"""BC/PPOで共有するHydraモデル設定の合成テスト。"""

from pathlib import Path

from hydra import compose, initialize_config_dir

from kaggriculture.policy.common.config import ModelConfig

_CONFIG_DIR = Path(__file__).parents[2] / "src/kaggriculture/training/conf"


def _compose(name: str, overrides: list[str] | None = None):
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR.resolve())):
        return compose(config_name=name, overrides=overrides or [])


def test_bc_and_ppo_share_actor_structure() -> None:
    bc = _compose("bc")
    ppo = _compose("ppo")

    actor_fields = (
        "d_model",
        "num_heads",
        "d_feedforward",
        "num_layers_encoder",
        "num_layers_decoder",
        "num_layers_critic",
    )
    assert all(bc.model[name] == ppo.model[name] for name in actor_fields)
    ModelConfig(**dict(bc.model))
    ModelConfig(**dict(ppo.model))


def test_cli_overrides_apply_after_shared_model_config() -> None:
    cfg = _compose("ppo", ["model.d_model=64", "experiment.name=small"])

    assert cfg.model.d_model == 64
    assert cfg.experiment.name == "small"


def test_ppo_stability_defaults() -> None:
    cfg = _compose("ppo")

    assert cfg.ppo.learning_rate == 1.0e-5
    assert cfg.ppo.full_game_rounds == 2
    assert cfg.ppo.rollout_horizon == cfg.rules.episode_steps
    assert cfg.ppo.update_epochs == 1
    assert cfg.ppo.daily_reward_coefficient == 0.0
    assert cfg.ppo.entropy_coef == 0.0
    assert cfg.ppo.target_kl == 0.02
    assert cfg.ppo.anchor_sample_prob == 0.25
    assert cfg.ppo.pool_sample_prob == 0.5
    assert cfg.ppo.eval_episodes == 64
    assert cfg.ppo.promotion_win_rate == 0.55
    assert cfg.ppo.pool_size == 8
    assert cfg.ppo.anchor_promotion_win_rate == 0.6
    assert cfg.ppo.reference_actor_l2_coef == 0.1
    assert cfg.ppo.reference_checkpoint is None
    assert list(cfg.ppo.matched_opponents) == []


def test_value_pretraining_reward_matches_ppo_default() -> None:
    value = _compose("value_pretrain")
    ppo = _compose("ppo")

    assert value.train.gamma == ppo.ppo.gamma
    assert value.train.daily_reward_coefficient == ppo.ppo.daily_reward_coefficient
    assert value.train.daily_reward_scale == ppo.ppo.daily_reward_scale
    assert value.train.daily_reward_maximum == ppo.ppo.daily_reward_maximum


def test_diverse_mmpq_ppo_pairs_fixed_opponents_with_state_banks() -> None:
    cfg = _compose("ppo_mmpq_diverse")

    assert cfg.ppo.rollout_horizon == 456
    assert cfg.ppo.full_game_rounds == 2
    assert [entry.name for entry in cfg.ppo.matched_opponents] == [
        "dsm_bc",
        "decem_bc",
        "decem_ppo_260",
    ]
    assert cfg.ppo.matched_opponents[0].opening_checkpoint.endswith(
        "opening_day10_dsm_v1/2026-09-29/10-05-16/checkpoints/best"
    )
    assert all(entry.state_bank_path for entry in cfg.ppo.matched_opponents)


def test_focused_diverse_ppo_pairs_each_opponent_with_its_opening() -> None:
    cfg = _compose("ppo_mmpq_focused_diverse")

    assert cfg.ppo.rollout_horizon == 456
    assert [entry.name for entry in cfg.ppo.matched_opponents] == [
        "mmpq_focused",
        "decem_focused",
        "dsm_focused",
        "vadim_focused",
    ]
    assert cfg.ppo.init_bc_checkpoint.endswith(
        "day11plus_mmpq_ppo_light_v1/2026-09-29/22-33-21/checkpoints/best"
    )
    assert all(entry.opening_checkpoint for entry in cfg.ppo.matched_opponents)
    assert all(entry.state_bank_path for entry in cfg.ppo.matched_opponents)
