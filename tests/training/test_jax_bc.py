"""固定shape cacheとJAX BC更新の契約テスト。"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import actions as A
from kaggriculture.policy.jax import distribution as D
from kaggriculture.policy.jax import features as F
from kaggriculture.policy.jax import model as M
from kaggriculture.policy.jax import tokenize as JT
from kaggriculture.policy.torch import tokenize as TT
from kaggriculture.rules import constants as C
from kaggriculture.simulator import market
from kaggriculture.training.bc import jax_core
from kaggriculture.training.bc.jax_cache import (
    _MARKET_CHOICE,
    _UNIT_CHOICE,
    BCBatch,
    CacheRules,
    _load_cache_stats,
    _trace_choices,
    load_shard,
    observation_to_state,
    prepare_episode,
    prepare_episodes,
)
from tests.policy.conftest import make_fresh_observation


def _aggregate(index: np.ndarray, value: np.ndarray) -> np.ndarray:
    result = np.zeros((index.shape[0], V.VOCAB_SIZE), dtype=np.float32)
    rows = np.broadcast_to(np.arange(index.shape[0])[:, None], index.shape)
    np.add.at(result, (rows, index), value)
    return result


def _observation(player: int = 0) -> dict:
    obs = make_fresh_observation(player)
    inventory = jnp.asarray([obs["market"]["inventory"][item] for item in C.PRODUCTS])
    prices = np.asarray(market.market_price(inventory))
    obs["market"]["prices"] = dict(zip(C.PRODUCTS, (int(price) for price in prices), strict=True))
    return obs


def _small_model() -> tuple[M.PolicyValueNet, dict, ModelConfig]:
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
    model = M.PolicyValueNet(config)
    return model, M.initialize(model, jax.random.key(0)), config


def test_cached_candidate_tables_match_jax_order() -> None:
    unit = {
        (int(op), int(arg)): i
        for i, (op, arg) in enumerate(zip(A.UNIT_CANDIDATES.op, A.UNIT_CANDIDATES.arg, strict=True))
    }
    market_choices = {
        (int(op), int(arg)): i
        for i, (op, arg) in enumerate(
            zip(A.MARKET_CANDIDATES.op, A.MARKET_CANDIDATES.arg, strict=True)
        )
    }
    assert _UNIT_CHOICE == unit
    assert _MARKET_CHOICE == market_choices


def test_observation_state_tokenization_matches_torch() -> None:
    obs = _observation()
    state = observation_to_state(obs)
    expected = F.pack_vectors(TT.get_encoder_input(obs), F.MAX_ENCODER_FEATURES)
    actual = JT.encode_observation(state, jnp.asarray(obs["player"]))
    expected_dense = _aggregate(expected.index, expected.value)
    actual_dense = _aggregate(np.asarray(actual.index), np.asarray(actual.value))
    np.testing.assert_allclose(actual_dense, expected_dense, atol=1e-6)


def test_trace_choices_is_accepted_by_jax_teacher_forcing() -> None:
    obs = _observation()
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    choices, mask = _trace_choices(obs, action, CacheRules())
    state = observation_to_state(obs)
    batch_state = jax.tree.map(lambda value: jnp.asarray(value)[None], state)
    model, variables, _ = _small_model()
    result = D.evaluate_choices(
        model,
        variables,
        batch_state,
        jnp.asarray([0]),
        jnp.asarray(choices[None]),
    )
    assert int(result.num_decisions[0]) == int(mask.sum()) == 2
    assert bool(jnp.isfinite(result.log_prob[0]))


def test_episode_cache_round_trip(tmp_path: Path) -> None:
    obs = _observation()
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    episode = tmp_path / "episode.json"
    episode.write_text(
        json.dumps(
            {
                "rewards": [1.0],
                "steps": [
                    [{"observation": obs, "action": None}],
                    [{"observation": obs, "action": action}],
                ],
            }
        )
    )
    path = prepare_episode(episode, tmp_path / "cache", CacheRules())
    assert path is not None
    batch = load_shard(path)
    assert batch.players.tolist() == [0]
    assert batch.choices.shape == (1, L.MAX_DECODE_LEN)
    assert int(batch.decision_mask.sum()) == 2
    assert prepare_episode(episode, tmp_path / "cache", CacheRules()) == path
    stats = _load_cache_stats(path)
    assert stats.accepted == 1
    assert stats.discarded == 0


def test_episode_cache_can_select_one_player(tmp_path: Path) -> None:
    observations = [_observation(0), _observation(1)]
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    episode = tmp_path / "episode.json"
    episode.write_text(
        json.dumps(
            {
                "rewards": [10.0, 20.0],
                "steps": [
                    [
                        {"observation": observations[0], "action": None},
                        {"observation": observations[1], "action": None},
                    ],
                    [
                        {"observation": observations[0], "action": action},
                        {"observation": observations[1], "action": action},
                    ],
                ],
            }
        )
    )

    path = prepare_episode(episode, tmp_path / "cache", CacheRules(), (1,))
    batch = load_shard(path)

    assert batch.players.tolist() == [1]


def test_episode_cache_reports_discarded_samples(tmp_path: Path) -> None:
    obs = _observation()
    episode = tmp_path / "episode.json"
    episode.write_text(
        json.dumps(
            {
                "rewards": [1.0],
                "steps": [
                    [{"observation": {"player": 0}, "action": None}],
                    [
                        {
                            "observation": obs,
                            "action": {"farmer": ["PASS"], "hands": [], "market": []},
                        }
                    ],
                ],
            }
        )
    )

    paths, created, stats = prepare_episodes([episode], tmp_path / "cache", CacheRules())

    assert paths == []
    assert created == 0
    assert stats.accepted == 0
    assert stats.discarded == 1
    assert sum(stats.reasons.values()) == 1
    sidecar = next((tmp_path / "cache").glob("*.stats.json"))
    persisted = json.loads(sidecar.read_text())
    assert persisted["discarded"] == 1


def test_jax_bc_update_changes_actor_parameters() -> None:
    obs = _observation()
    choices, mask = _trace_choices(
        obs, {"farmer": ["PASS"], "hands": [], "market": []}, CacheRules()
    )
    state = observation_to_state(obs)
    batch = BCBatch(
        jax.tree.map(lambda value: jnp.asarray(value)[None], state),
        jnp.asarray([0]),
        jnp.asarray(choices[None]),
        jnp.asarray(mask[None]),
    )
    model, variables, _ = _small_model()
    config = jax_core.BCConfig(1e-3, 0.0, 1.0, 24, 100, 1.0)
    train_state = jax_core.create_train_state(model, variables, config)
    updated, metrics = jax_core.update_minibatch(model, train_state, batch, config)
    before = jax.tree.leaves(train_state.params)
    after = jax.tree.leaves(updated.params)
    assert bool(jnp.isfinite(metrics.loss))
    assert any(not np.array_equal(a, b) for a, b in zip(before, after, strict=True))
