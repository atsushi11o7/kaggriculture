"""完了試合を蓄積して1回だけ更新するPPO(full_game)の境界条件。"""

from dataclasses import replace

import jax
import jax.numpy as jnp

from kaggriculture.policy.jax import history as H
from kaggriculture.policy.jax import model as M
from kaggriculture.policy.jax import policy as P
from kaggriculture.training.ppo import core, full_game
from kaggriculture.training.ppo.rollout import RolloutConfig
from tests.policy.test_policy import _model

ENVS = 1
EPISODE_STEPS = 5  # 4ターンの軽量な契約テスト


def _config(**changes) -> RolloutConfig:
    return replace(
        RolloutConfig(episode_steps=EPISODE_STEPS, daily_reward_coefficient=0.0), **changes
    )


def _collect(model, variables, seat=0, config=None):
    config = config or _config()
    return config, full_game.collect_compact_game(
        model,
        variables,
        variables,
        seat,
        config,
        ENVS,
        jax.random.key(1),
        jax.random.key(2),
    )


def test_replayed_states_reach_the_recorded_final_cash() -> None:
    model, variables = _model()
    config, game = _collect(model, variables)
    state = full_game.initial_state(jax.random.key(1), ENVS, config)

    final = state
    counters = H.zeros(ENVS)
    replayed_steps = []
    for start, end in full_game.segment_bounds(game.dones.shape[0], 5):
        window = full_game._window(start, end)
        (final, counters), (states, _) = full_game.replay_segment(
            config, final, counters, window(game.actions)
        )
        replayed_steps.extend(states.step[:, 0].tolist())

    assert replayed_steps == list(range(EPISODE_STEPS - 1))
    assert jnp.array_equal(final.money, game.final_cash)
    assert bool(game.dones[-1].all()) and not bool(game.dones[:-1].any())


def test_targets_propagate_the_terminal_result_from_the_last_turn() -> None:
    rewards = jnp.asarray([[0.0], [0.0], [1.0]])
    game = full_game.CompactGame(
        *(None,) * 4,
        jnp.zeros((3, 1)),
        rewards,
        *(None,) * 1,
        jnp.asarray([[False], [False], [True]]),
        *(None,) * 5,
    )

    advantages, returns = full_game.game_targets(game, gamma=0.5, gae_lambda=1.0, value_lambda=1.0)

    assert jnp.allclose(returns[:, 0], jnp.asarray([0.25, 0.5, 1.0]))
    assert jnp.allclose(advantages[:, 0], returns[:, 0])
    # 方策側のλを小さくしても、価値の教師は実際のreturn(λ=1)のまま
    small, same_returns = full_game.game_targets(game, 0.5, 0.0, 1.0)
    assert jnp.allclose(same_returns[:, 0], returns[:, 0])
    assert jnp.allclose(small[:, 0], jnp.asarray([0.0, 0.0, 1.0]))


def test_accumulated_gradient_matches_a_single_batch_gradient() -> None:
    """区間分割とminibatch化を経ても、単一batchでの直接計算と一致することを確認する。

    GPUの既定の行列積精度(TF32相当)は、同じ計算でもbatch形状によって丸め方が変わり、
    1e-4を超える差が出ることがある(別途、batch不変性の実験で確認済み)。ここでは、
    数値的な正しさそのものを見たいので、最大精度に固定して比較する。
    """
    with jax.default_matmul_precision("highest"):
        _run_accumulated_gradient_check()


def _run_accumulated_gradient_check() -> None:
    model, variables = _model()
    config, game = _collect(model, variables)
    advantages, returns = full_game.game_targets(game, 0.999, 0.95, 1.0)
    state = full_game.initial_state(jax.random.key(1), ENVS, config)
    steps = game.dones.shape[0]
    counters = H.zeros(ENVS)
    (state, counters), (states, state_counters) = full_game.replay_segment(
        config, state, counters, game.actions
    )
    window = full_game._window(0, steps)
    rows = {
        "intent": window(game.intent),
        "slot_mask": game.slot_mask,
        "old_slot_log_prob": game.slot_log_prob,
        "old_value": game.value,
        "advantages": (advantages - advantages.mean()) / (advantages.std() + 1e-8),
        "returns": returns,
        "counters": jax.tree.map(lambda value: value[:, :, 0], state_counters),
    }
    ppo_config = core.PPOConfig(normalize_advantages=False, entropy_coef=0.0)
    minibatch = 2  # 4行を2つに分ける
    total, used = steps * ENVS, (steps * ENVS // minibatch) * minibatch

    accumulated, _, count = full_game.segment_gradient(
        model,
        variables["params"],
        None,
        states,
        rows,
        ppo_config,
        minibatch,
        0,
        jax.random.key(3),
    )

    order = jax.random.permutation(jax.random.key(3), total)[:used]

    def flat(value):
        return value.reshape((total,) + value.shape[2:])[order]

    batch = core.PPOBatch(
        jax.tree.map(flat, states),
        jnp.zeros((used,), jnp.int32),
        jax.tree.map(flat, rows["intent"]),
        flat(rows["slot_mask"]),
        flat(rows["old_slot_log_prob"]),
        flat(rows["old_value"]),
        flat(rows["advantages"]),
        flat(rows["returns"]),
        jax.tree.map(flat, rows["counters"]),
    )
    expected = jax.grad(lambda p: core._loss(model, p, batch, ppo_config)[0])(variables["params"])

    mean = jax.tree.map(lambda x: x / count, accumulated)
    differences = jax.tree.leaves(jax.tree.map(lambda a, b: jnp.abs(a - b).max(), mean, expected))
    assert count == used // minibatch
    assert max(float(value) for value in differences) < 1e-4


def test_full_game_update_alternates_seats_and_changes_parameters() -> None:
    model, variables = _model()
    config = _config()
    ppo_config = core.PPOConfig(normalize_advantages=False, entropy_coef=0.0, learning_rate=1e-3)
    train_state = core.create_train_state(model, variables, ppo_config)

    new_state, diagnostics = full_game.full_game_update(
        model,
        train_state,
        None,
        ppo_config,
        config,
        [variables, variables],
        jax.random.key(5),
        batch_size=ENVS,
        minibatch=2,
        segment_length=2,
        gamma=0.999,
        gae_lambda=0.95,
        value_lambda=1.0,
    )

    changed = jax.tree.leaves(
        jax.tree.map(lambda a, b: bool(jnp.any(a != b)), train_state.params, new_state.params)
    )
    assert any(changed)
    assert diagnostics["games"] == 2 * ENVS
    assert abs(diagnostics["win"] + diagnostics["lose"] + diagnostics["draw"] - 1.0) < 1e-6
    assert -1.0 <= diagnostics["round_gradient_cosine"] <= 1.0
    assert jnp.isfinite(diagnostics["loss"]) and diagnostics["gradient_norm"] > 0
    _ = M, P
