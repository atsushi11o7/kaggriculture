"""日次資産報酬とBC actor保持の境界条件。"""

from dataclasses import asdict, replace

import jax
import jax.numpy as jnp
import pytest
from flax.core import freeze, unfreeze

from kaggriculture.policy.common.config import checkpoint_shape_metadata
from kaggriculture.policy.jax import history as H
from kaggriculture.policy.jax import model as M
from kaggriculture.policy.jax import policy as P
from kaggriculture.rules import game_params as G
from kaggriculture.simulator.reset import reset
from kaggriculture.training.bc import core as bc_core
from kaggriculture.training.checkpoint import save_checkpoint
from kaggriculture.training.ppo import core
from kaggriculture.training.ppo.rollout import RolloutConfig, collect_rollout, to_ppo_batch
from kaggriculture.training.ppo.train import _load_actor_checkpoint
from kaggriculture.training.rl import DailyRewardConfig, daily_asset_rewards, estimated_assets
from tests.policy.test_policy import _model


def test_daily_reward_counts_purchases_as_assets_and_hires_as_daily_cost() -> None:
    state = reset(jax.random.key(0), 1)
    initial = estimated_assets(state)
    crop = 0
    bought = state._replace(
        money=state.money.at[0, 0].add(-G.CROP_SEED_COST[crop]),
        seeds=state.seeds.at[0, 0, crop].set(1),
        step=state.step.at[0].set(24),
    )
    reward, margin = daily_asset_rewards(
        bought, jnp.zeros(1), jnp.asarray([False]), 24, DailyRewardConfig(1.0, 100.0, 1.0)
    )
    assert jnp.allclose(estimated_assets(bought), initial)
    assert jnp.allclose(reward, 0)
    assert jnp.allclose(margin, 0)

    hired = bought._replace(
        money=bought.money.at[0, 0].add(-5),
        hands_active=bought.hands_active.at[0, 0, 0].set(True),
        step=bought.step.at[0].set(25),
    )
    # 日中はまだ評価せず、その日の収益と雇用費を日末にまとめて比較する。
    intraday, unchanged = daily_asset_rewards(
        hired, margin, jnp.asarray([False]), 24, DailyRewardConfig(1.0, 100.0, 1.0)
    )
    assert jnp.allclose(intraday, 0)
    assert jnp.allclose(unchanged, margin)
    end = hired._replace(step=hired.step.at[0].set(48))
    end_reward, _ = daily_asset_rewards(
        end, margin, jnp.asarray([False]), 24, DailyRewardConfig(1.0, 100.0, 1.0)
    )
    assert jnp.allclose(end_reward, jnp.asarray([[-0.05, 0.05]]))

    # 最終日の途中で試合が終わっても、未評価の利益・損失を落とさない。
    terminal, _ = daily_asset_rewards(
        hired, margin, jnp.asarray([True]), 24, DailyRewardConfig(1.0, 100.0, 1.0)
    )
    assert jnp.allclose(terminal, jnp.asarray([[-0.05, 0.05]]))


def test_inventory_value_cannot_be_inflated_by_market_inventory_changes() -> None:
    state = reset(jax.random.key(0), 1)
    wheat = 0
    held = state._replace(shed=state.shed.at[0, 0, wheat].set(10))
    scarcer_market = held._replace(market_inventory=held.market_inventory.at[0, wheat].add(-500))
    assert jnp.allclose(estimated_assets(held), estimated_assets(scarcer_market))


def test_bc_actor_penalty_ignores_critic_only_parameters() -> None:
    model, variables = _model()
    reference = variables["params"]
    assert jnp.allclose(core._reference_actor_l2(reference, reference), 0)

    changed = unfreeze(reference)
    changed["policy_proj"]["kernel"] += 0.1
    assert core._reference_actor_l2(freeze(changed), reference) > 0

    critic_changed = unfreeze(reference)
    critic_changed["value_head"]["layers_0"]["kernel"] += 0.1
    assert jnp.allclose(core._reference_actor_l2(freeze(critic_changed), reference), 0)


def test_ppo_update_accepts_bc_actor_anchor() -> None:
    model, variables = _model()
    rollout = collect_rollout(
        model,
        variables,
        RolloutConfig(horizon=1),
        reset(jax.random.key(1), 1),
        H.zeros(1),
        jax.random.key(2),
    )
    batch = to_ppo_batch(rollout)
    config = core.PPOConfig(reference_actor_l2_coef=0.1)
    state = core.create_train_state(model, variables, config)
    changed = unfreeze(state.params)
    changed["policy_proj"]["kernel"] += 0.01
    state = state.replace(params=changed)
    updated, metrics = core.update_minibatch(model, state, batch, config, variables["params"])
    assert updated.step == 1
    assert jnp.isfinite(metrics.loss)
    assert metrics.reference_actor_l2 > 0

    updated, metrics, epochs = core.update_epochs(
        model, state, batch, jax.random.key(3), config, 1, 2, variables["params"]
    )
    assert updated.step == 1
    assert epochs == 1
    assert jnp.isfinite(metrics.loss)


def test_daily_reward_config_rejects_invalid_scale() -> None:
    try:
        DailyRewardConfig(scale=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero scale must be rejected")


@pytest.mark.parametrize("trainer", ["bc", "ppo"])
def test_reference_actor_loads_bc_and_ppo_checkpoints(tmp_path, trainer: str) -> None:
    model, source_variables = _model()
    target_model = M.PolicyValueNet(replace(model.config, use_asymmetric_critic=trainer == "bc"))
    target_variables = P.initialize(target_model, jax.random.key(7))
    source_state = (
        bc_core.create_train_state(model, source_variables, bc_core.BCConfig())
        if trainer == "bc"
        else core.create_train_state(model, source_variables, core.PPOConfig())
    )
    changed = unfreeze(source_state.params)
    changed["policy_proj"]["kernel"] += 0.1
    changed["value_head"]["layers_0"]["kernel"] += 0.1
    source_state = source_state.replace(params=freeze(changed))
    checkpoint = tmp_path / trainer
    save_checkpoint(
        checkpoint,
        source_state,
        {**checkpoint_shape_metadata(), "trainer": trainer, "model_config": asdict(model.config)},
    )

    loaded = _load_actor_checkpoint(checkpoint, target_variables, target_model.config)["params"]
    assert jnp.allclose(loaded["policy_proj"]["kernel"], changed["policy_proj"]["kernel"])
    assert jnp.allclose(
        loaded["value_head"]["layers_0"]["kernel"],
        target_variables["params"]["value_head"]["layers_0"]["kernel"],
    )


def test_reference_actor_rejects_incompatible_checkpoint(tmp_path) -> None:
    model, source_variables = _model()
    source_state = core.create_train_state(model, source_variables, core.PPOConfig())
    checkpoint = tmp_path / "source"
    save_checkpoint(
        checkpoint,
        source_state,
        {**checkpoint_shape_metadata(), "trainer": "ppo", "model_config": asdict(model.config)},
    )
    target = unfreeze(source_variables["params"])
    target["policy_proj"]["kernel"] = jnp.zeros((1, 1))
    with pytest.raises(ValueError, match="incompatible actor checkpoint"):
        _load_actor_checkpoint(checkpoint, {"params": freeze(target)}, model.config)


@pytest.mark.parametrize("override", [{"num_heads": 4}, {"use_episode_history": True}])
def test_reference_actor_rejects_semantically_incompatible_config(tmp_path, override) -> None:
    model, source_variables = _model()
    source_state = core.create_train_state(model, source_variables, core.PPOConfig())
    checkpoint = tmp_path / "source"
    save_checkpoint(
        checkpoint,
        source_state,
        {**checkpoint_shape_metadata(), "trainer": "ppo", "model_config": asdict(model.config)},
    )
    target_config = replace(model.config, **override)
    with pytest.raises(ValueError, match="incompatible actor checkpoint config"):
        _load_actor_checkpoint(checkpoint, source_variables, target_config)
