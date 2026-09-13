"""JAX自己回帰方策とシミュレータ接続の契約テスト。"""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import distribution as D
from kaggriculture.policy.jax import model as JM
from kaggriculture.policy.torch import model as TM
from kaggriculture.simulator.reset import reset
from kaggriculture.simulator.step import step_batch_lockstep
from kaggriculture.training.weight_bridge import torch_to_jax


def _policy():
    config = ModelConfig(
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
    torch.manual_seed(0)
    reference = TM.PolicyValueNet(config).eval()
    model = JM.PolicyValueNet(config)
    return model, torch_to_jax(reference, config)


def test_sample_and_teacher_forcing_log_prob_match() -> None:
    """増分生成と一括教師強制が同じ複合行動確率を返す。"""
    model, variables = _policy()
    states = reset(jax.random.key(1), batch_size=1)
    cache = D.init_decode_cache(model, jax.random.key(2), batch_size=2)
    sample = jax.jit(
        lambda state, key: D.sample_self_play_actions(
            model, variables, cache, state, key, greedy=False
        )
    )(states, jax.random.key(3))

    doubled_states = jax.tree.map(lambda value: jnp.concatenate([value, value]), states)
    players = jnp.asarray([0, 1], dtype=jnp.int32)
    choices = sample.choices.reshape(2, sample.choices.shape[-1])
    evaluated = jax.jit(
        lambda state, player, choice: D.evaluate_choices(model, variables, state, player, choice)
    )(doubled_states, players, choices)

    np.testing.assert_allclose(evaluated.value, sample.value.reshape(-1), atol=1e-6)
    # Incremental KV decodeと一括decodeではfloat32の演算順が異なる。
    np.testing.assert_allclose(evaluated.log_prob, sample.log_prob.reshape(-1), atol=2e-3)
    np.testing.assert_array_equal(evaluated.num_decisions, sample.num_decisions.reshape(-1))
    np.testing.assert_allclose(evaluated.token_log_prob.sum(axis=-1), evaluated.log_prob, atol=1e-6)
    np.testing.assert_allclose(sample.token_log_prob.sum(axis=-1), sample.log_prob, atol=5e-6)


def test_generated_actions_step_jax_simulator() -> None:
    """両者の生成行動をCPUへ戻さずJAXシミュレータへ渡せる。"""
    model, variables = _policy()
    states = reset(jax.random.key(4), batch_size=2)
    cache = D.init_decode_cache(model, jax.random.key(5), batch_size=4)

    @jax.jit
    def generate_and_step(state, key):
        output = D.sample_self_play_actions(model, variables, cache, state, key, greedy=True)
        next_state, reward, done = step_batch_lockstep(state, output.action)
        return next_state, reward, done

    next_state, reward, done = generate_and_step(states, jax.random.key(6))
    np.testing.assert_array_equal(next_state.step, [1, 1])
    assert reward.shape == (2, 2)
    assert done.shape == (2,)
