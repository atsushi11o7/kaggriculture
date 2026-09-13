"""JAX BC Hydra設定の合成テスト。"""

from pathlib import Path

from hydra import compose, initialize_config_dir

from kaggriculture.policy.common.config import ModelConfig

_CONFIG_DIR = Path(__file__).parents[2] / "src/kaggriculture/training/conf"


def test_jax_bc_and_ppo_share_actor_structure() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR.resolve())):
        bc = compose(config_name="bc_jax")
        ppo = compose(config_name="ppo")
    fields = (
        "d_model",
        "num_heads",
        "d_feedforward",
        "num_layers_encoder",
        "num_layers_decoder",
        "dropout",
        "use_episode_history",
    )
    assert all(bc.model[name] == ppo.model[name] for name in fields)
    assert not bc.model.use_asymmetric_critic
    assert ppo.model.use_asymmetric_critic
    ModelConfig(**dict(bc.model))


def test_default_training_profiles_are_bounded() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR.resolve())):
        bc = compose(config_name="bc_jax")
        ppo = compose(config_name="ppo")

    assert bc.data.data_dir == "data/replays"
    assert bc.data.num_episodes == 256
    assert bc.train.batch_size == 64
    assert ppo.env.batch_size * ppo.ppo.rollout_horizon * 2 % ppo.ppo.minibatch_size == 0
    assert ppo.ppo.learning_rate <= bc.train.learning_rate
    assert ppo.ppo.eval_interval > 0
