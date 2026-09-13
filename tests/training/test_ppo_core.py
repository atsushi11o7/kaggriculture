"""JAX PPOの報酬・GAE・更新処理のテスト。"""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import distribution as D
from kaggriculture.policy.jax import model as JM
from kaggriculture.policy.torch import model as TM
from kaggriculture.simulator.reset import reset
from kaggriculture.training.ppo import core
from kaggriculture.training.weight_bridge import torch_to_jax


def _policy():
    model_config = ModelConfig(
        d_model=16,
        num_heads=4,
        d_feedforward=32,
        num_layers_encoder=1,
        num_layers_decoder=1,
        dropout=0.0,
        use_episode_history=False,
        use_asymmetric_critic=False,
        num_layers_critic=1,
    )
    torch.manual_seed(1)
    reference = TM.PolicyValueNet(model_config).eval()
    model = JM.PolicyValueNet(model_config)
    return model, torch_to_jax(reference, model_config)


def test_terminal_win_rewards_are_zero_sum() -> None:
    cash = jnp.asarray([[100.0, 20.0], [5.0, 5.0], [1.0, 3.0]])
    done = jnp.asarray([True, True, False])
    expected = np.asarray([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(core.terminal_win_rewards(cash, done), expected)


def test_compute_gae_respects_terminal_boundary() -> None:
    rewards = jnp.asarray([[[0.0]], [[1.0]], [[5.0]]])
    values = jnp.zeros_like(rewards)
    dones = jnp.asarray([[False], [True], [False]])
    advantage, returns = core.compute_gae(
        rewards, values, dones, jnp.asarray([[2.0]]), gamma=1.0, gae_lambda=1.0
    )
    expected = np.asarray([[[1.0]], [[1.0]], [[7.0]]])
    np.testing.assert_allclose(advantage, expected)
    np.testing.assert_allclose(returns, expected)


def test_update_minibatch_changes_finite_parameters() -> None:
    model, variables = _policy()
    config = core.PPOConfig(learning_rate=1e-4)
    train_state = core.create_train_state(model, variables, config)
    states = reset(jax.random.key(10), batch_size=1)
    cache = D.init_decode_cache(model, jax.random.key(11), batch_size=2)
    sample = jax.jit(
        lambda state, key: D.sample_self_play_actions(
            model, variables, cache, state, key, greedy=False
        )
    )(states, jax.random.key(12))
    doubled_states = jax.tree.map(lambda value: jnp.concatenate([value, value]), states)
    batch = core.PPOBatch(
        states=doubled_states,
        players=jnp.asarray([0, 1], dtype=jnp.int32),
        choices=sample.choices.reshape(2, sample.choices.shape[-1]),
        decision_mask=sample.decision_mask.reshape(2, sample.decision_mask.shape[-1]),
        old_token_log_prob=sample.token_log_prob.reshape(2, sample.token_log_prob.shape[-1]),
        old_log_prob=sample.log_prob.reshape(-1),
        old_value=sample.value.reshape(-1),
        advantages=jnp.asarray([1.0, -1.0]),
        returns=sample.value.reshape(-1) + jnp.asarray([0.1, -0.1]),
    )

    updated, metrics = core.update_minibatch(model, train_state, batch, config)
    leaves_before = jax.tree.leaves(train_state.params)
    leaves_after = jax.tree.leaves(updated.params)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves_after)
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(leaves_before, leaves_after, strict=True)
    )
    assert bool(jnp.isfinite(metrics.loss))
