"""固定slot BC/PPOの最小学習経路。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from kaggriculture.policy.common.config import (
    ModelConfig,
    checkpoint_shape_metadata,
    validate_checkpoint_metadata,
)
from kaggriculture.policy.jax import history as H
from kaggriculture.policy.jax import model as M
from kaggriculture.policy.jax import policy as P
from kaggriculture.rules import constants as C
from kaggriculture.simulator.reset import reset
from kaggriculture.training.bc import core as bc_core
from kaggriculture.training.bc.dataset import action_to_intent, stack_samples
from kaggriculture.training.bc.train import _restored_best, _schedule_steps
from kaggriculture.training.ppo import core as ppo_core
from kaggriculture.training.ppo.rollout import (
    RolloutConfig,
    collect_rollout,
    collect_rollout_vs_opponent,
    to_ppo_batch,
)
from kaggriculture.training.ppo.train import (
    _evaluation_summary,
    _pool_members,
    _select_opponent,
)
from kaggriculture.training.replays.state import CacheRules
from tests.policy.conftest import make_fresh_observation


def _model():
    config = ModelConfig(8, 1, 16, 1, 1, 0.0, False, False, 1)
    model = M.PolicyValueNet(config)
    return model, P.initialize(model, jax.random.key(0))


def test_expert_action_maps_to_fixed_slots() -> None:
    obs = make_fresh_observation()
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    intent, mask = action_to_intent(obs, action, CacheRules())

    assert intent.unit.shape == (C.MAX_HANDS + 1,)
    assert intent.market.shape == (10,)
    assert mask.sum() == 2  # farmerと市場STOP


def test_bc_update() -> None:
    model, variables = _model()
    obs = make_fresh_observation()
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    intent, mask = action_to_intent(obs, action, CacheRules())
    from kaggriculture.training.replays.state import observation_to_state

    batch = stack_samples([(observation_to_state(obs), 0, intent, mask)])
    train_state = bc_core.create_train_state(model, variables, bc_core.BCConfig())
    train_state, (loss, count, excluded) = bc_core.update_minibatch(
        model, train_state, jax.device_put(batch), bc_core.BCConfig()
    )

    assert jnp.isfinite(loss)
    assert count == 2
    assert excluded == 0


def test_bc_loss_excludes_structurally_illegal_labels() -> None:
    """固定mask上で不可能な有効ラベルをlossから除外する。"""
    from kaggriculture.policy.jax import actions as A
    from kaggriculture.training.replays.state import observation_to_state

    model, variables = _model()
    obs = make_fresh_observation()
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    intent, mask = action_to_intent(obs, action, CacheRules())

    # obsのseedsは全crop 0なので、PLANTはshedと無関係に常に不合法。
    (candidates,) = jnp.nonzero(A.UNIT_CANDIDATES.op == C.FARMER_OP_PLANT)
    unit = intent.unit.copy()
    unit[0] = int(candidates[0])  # farmer slotへ不合法なPLANTを差し替える
    unit[1] = int(candidates[0])  # inactive handは不合法でも集計対象外
    intent = intent._replace(unit=unit)

    batch = stack_samples([(observation_to_state(obs), 0, intent, mask)])
    train_state = bc_core.create_train_state(model, variables, bc_core.BCConfig())
    train_state, (loss, count, excluded) = bc_core.update_minibatch(
        model, train_state, jax.device_put(batch), bc_core.BCConfig()
    )

    assert excluded == 1
    assert count == 1  # farmer slot(illegal)が除外され、市場STOP slotだけ残る
    assert jnp.isfinite(loss)


def test_ppo_rollout_and_update() -> None:
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
    config = ppo_core.PPOConfig()
    train_state = ppo_core.create_train_state(model, variables, config)
    _, metrics = ppo_core.update_minibatch(model, train_state, batch, config)

    assert batch.players.shape == (2,)
    assert jnp.isfinite(metrics.loss)


def test_ppo_target_kl_stops_after_first_epoch() -> None:
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
    config = ppo_core.PPOConfig(target_kl=-1.0)
    train_state = ppo_core.create_train_state(model, variables, config)
    key = jax.random.key(3)

    stopped, metrics, epochs = ppo_core.update_epochs(model, train_state, batch, key, config, 3, 2)
    first, first_metrics, first_epochs = ppo_core.update_epochs(
        model, train_state, batch, key, replace(config, target_kl=None), 1, 2
    )

    assert epochs == first_epochs == 1
    assert stopped.step == first.step == 1
    assert jnp.isfinite(metrics.loss)
    assert jnp.allclose(metrics.loss, first_metrics.loss)


def test_collect_rollout_vs_opponent_masks_opponent_slots() -> None:
    model, learner_variables = _model()
    _, opponent_variables = _model()
    rollout = collect_rollout_vs_opponent(
        model,
        learner_variables,
        opponent_variables,
        0,
        RolloutConfig(horizon=2),
        reset(jax.random.key(1), 3),
        H.zeros(3),
        jax.random.key(2),
    )
    batch = to_ppo_batch(rollout)

    # learner_seat=0の行(players==0)だけが損失に寄与し、
    # opponent側(players==1)は全slotがmaskされている。
    learner_rows = batch.players == 0
    opponent_rows = batch.players == 1
    assert bool(jnp.any(batch.slot_mask[learner_rows]))
    assert not bool(jnp.any(batch.slot_mask[opponent_rows]))
    assert jnp.isfinite(rollout.value).all()


def test_advantage_normalization_ignores_fixed_opponent_rows() -> None:
    mask = jnp.asarray([True, True, False, False])
    baseline = ppo_core._normalize_advantages(jnp.asarray([1.0, 3.0, 10.0, -10.0]), mask)
    changed = ppo_core._normalize_advantages(jnp.asarray([1.0, 3.0, 1e6, -1e6]), mask)

    assert jnp.allclose(baseline[:2], changed[:2])
    assert jnp.allclose(baseline[:2], jnp.asarray([-1.0, 1.0]))


def test_bc_resume_restores_validation_best() -> None:
    assert _restored_best({"validation_loss": 0.25}) == 0.25
    assert _restored_best({}) == float("inf")
    assert _restored_best({"best_validation_loss": 0.2, "validation_loss": 0.3}) == 0.2


def test_select_opponent_uses_anchor_pool_and_self_probabilities() -> None:
    assert _select_opponent(0.1, True, 0.5, 0.25) == "anchor"
    assert _select_opponent(0.6, True, 0.5, 0.25) == "pool"
    assert _select_opponent(0.9, True, 0.5, 0.25) == "self"


def test_select_opponent_routes_empty_pool_share_to_anchor() -> None:
    # poolが空の間は、pool分の確率もself-playへ逃がさずanchorへ回す。
    assert _select_opponent(0.6, False, 0.5, 0.25) == "anchor"
    assert _select_opponent(0.8, False, 0.5, 0.25) == "self"


def test_evaluation_summary_reports_outcomes_seats_and_cash() -> None:
    from kaggriculture.training.ppo.evaluation import EvaluationResult

    result = EvaluationResult(
        cash=jnp.asarray([[10.0, 2.0], [5.0, 5.0], [3.0, 9.0], [8.0, 4.0]]),
        outcome=jnp.asarray([1.0, 0.0, -1.0, 1.0]),
        win_rate=jnp.asarray(0.625),
    )

    summary = _evaluation_summary(result)

    assert summary == {
        "win_rate": 0.625,
        "wins": 2,
        "draws": 1,
        "losses": 1,
        "seat0_win_rate": 0.75,
        "seat1_win_rate": 0.5,
        "candidate_cash": 6.5,
        "opponent_cash": 5.0,
    }


def test_pool_members_are_sorted_by_numeric_suffix(tmp_path) -> None:
    for name in ("member_10", "member_2", "member_1", "notes"):
        (tmp_path / name).mkdir()

    assert [path.name for path in _pool_members(tmp_path)] == [
        "member_1",
        "member_2",
        "member_10",
    ]


def test_checkpoint_shape_metadata_is_self_consistent() -> None:
    metadata = checkpoint_shape_metadata()
    validate_checkpoint_metadata(metadata)
    assert metadata["max_hands"] == C.MAX_HANDS


def test_checkpoint_shape_metadata_rejects_different_hand_limit() -> None:
    metadata = {**checkpoint_shape_metadata(), "max_hands": 32}
    with pytest.raises(ValueError, match="incompatible policy checkpoint"):
        validate_checkpoint_metadata(metadata)


def test_checkpoint_shape_metadata_rejects_legacy_checkpoint() -> None:
    with pytest.raises(ValueError, match="incompatible policy checkpoint"):
        validate_checkpoint_metadata({"architecture": "fixed_slot", "trainer": "ppo"})


def test_bc_schedule_respects_step_limit() -> None:
    assert _schedule_steps(100, 20, -1) == 2000
    assert _schedule_steps(100, 20, 250) == 250
    with pytest.raises(ValueError, match="complete batch"):
        _schedule_steps(0, 20, -1)
    with pytest.raises(ValueError, match="max_epochs"):
        _schedule_steps(100, 0, -1)
