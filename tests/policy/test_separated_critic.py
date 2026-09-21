"""完全分離criticの互換初期化と勾配境界。"""

import jax
import jax.numpy as jnp
from flax import traverse_util
from flax.core import freeze, unfreeze

from kaggriculture.policy.common.config import CRITIC_PARAMETER_MODULES, ModelConfig
from kaggriculture.policy.jax import policy as P
from kaggriculture.policy.jax.model import PolicyValueNet
from kaggriculture.policy.jax.separated_model import SeparatedPolicyValueNet
from kaggriculture.simulator.reset import reset


def _config():
    return ModelConfig(8, 1, 16, 1, 1, 0.0, False, True, 1)


def _clone_shared_into_separated(shared, separated):
    source = traverse_util.flatten_dict(unfreeze(shared["params"]))
    target = traverse_util.flatten_dict(unfreeze(separated["params"]))
    aliases = {
        "critic_token_embedding": "token_embedding",
        "critic_board_position_embedding": "board_position_embedding",
        "critic_encoder": "encoder",
    }
    for name in target:
        source_name = name
        if source_name not in source and name[0] in aliases:
            source_name = (aliases[name[0]], *name[1:])
        target[name] = source[source_name]
    return {"params": freeze(traverse_util.unflatten_dict(target))}


def test_separated_critic_clones_shared_checkpoint_exactly() -> None:
    config = _config()
    shared_model = PolicyValueNet(config)
    separated_model = SeparatedPolicyValueNet(config)
    shared = P.initialize(shared_model, jax.random.key(0))
    separated = P.initialize(separated_model, jax.random.key(1))
    separated = _clone_shared_into_separated(shared, separated)
    states = reset(jax.random.key(2), 2)

    shared_values = P.state_values(shared_model, shared, states)
    separated_values = P.state_values(separated_model, separated, states)

    assert jnp.allclose(shared_values, separated_values, atol=1e-6)


def test_value_gradient_does_not_reach_separated_actor() -> None:
    model = SeparatedPolicyValueNet(_config())
    variables = P.initialize(model, jax.random.key(0))
    states = reset(jax.random.key(1), 1)

    gradients = jax.grad(lambda params: jnp.sum(P.state_values(model, {"params": params}, states)))(
        variables["params"]
    )

    actor_leaves = []
    critic_leaves = []
    for module, subtree in gradients.items():
        destination = critic_leaves if module in CRITIC_PARAMETER_MODULES else actor_leaves
        destination.extend(jax.tree.leaves(subtree))
    assert actor_leaves
    assert all(bool(jnp.all(gradient == 0)) for gradient in actor_leaves)
    assert any(bool(jnp.any(gradient != 0)) for gradient in critic_leaves)


def test_shared_value_checkpoint_migrates_to_separated_model(tmp_path) -> None:
    from dataclasses import asdict

    from kaggriculture.policy.common.config import (
        CRITIC_ARCHITECTURE_VERSION,
        checkpoint_shape_metadata,
    )
    from kaggriculture.training.checkpoint import save_checkpoint
    from kaggriculture.training.ppo import core as ppo_core
    from kaggriculture.training.ppo.train import _load_value_checkpoint

    config = _config()
    shared_model = PolicyValueNet(config)
    separated_model = SeparatedPolicyValueNet(config)
    shared = P.initialize(shared_model, jax.random.key(3))
    separated = P.initialize(separated_model, jax.random.key(4))
    checkpoint = tmp_path / "shared"
    save_checkpoint(
        checkpoint,
        ppo_core.create_train_state(shared_model, shared, ppo_core.PPOConfig()),
        {
            **checkpoint_shape_metadata(),
            "trainer": "value_pretrain",
            "critic_architecture_version": CRITIC_ARCHITECTURE_VERSION,
            "model_config": asdict(config),
            "reward_mode": "terminal_win",
            "gamma": 0.999,
        },
    )

    migrated = _load_value_checkpoint(checkpoint, separated, config, 0.999, 0.0, 10000.0, 0.02)
    states = reset(jax.random.key(5), 2)

    assert jnp.allclose(
        P.state_values(shared_model, shared, states),
        P.state_values(separated_model, migrated, states),
        atol=1e-6,
    )
