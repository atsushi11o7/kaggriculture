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
        "dropout",
        "use_episode_history",
    )
    assert all(bc.model[name] == ppo.model[name] for name in actor_fields)
    assert not bc.model.use_asymmetric_critic
    assert ppo.model.use_asymmetric_critic
    ModelConfig(**dict(bc.model))
    ModelConfig(**dict(ppo.model))


def test_cli_overrides_apply_after_shared_model_config() -> None:
    cfg = _compose("ppo", ["model.d_model=64", "experiment.name=small"])

    assert cfg.model.d_model == 64
    assert cfg.experiment.name == "small"


def test_ppo_stability_defaults() -> None:
    cfg = _compose("ppo")

    assert cfg.ppo.learning_rate == 5.0e-5
    assert cfg.ppo.entropy_coef == 0.0
    assert cfg.ppo.target_kl == 0.02
    assert cfg.ppo.anchor_sample_prob == 0.5
    assert cfg.ppo.pool_sample_prob == 0.5
    assert cfg.ppo.anchor_promotion_win_rate == 0.5
    assert cfg.ppo.reference_actor_l2_coef == 0.1
    assert cfg.ppo.reference_checkpoint is None
