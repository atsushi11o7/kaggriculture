"""Critic事前学習の凍結と教師値の基本性質。"""

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture.training.value_pretrain.core import create_train_state


class _Model:
    def apply(self, *_args, **_kwargs):
        return None


def test_critic_optimizer_keeps_actor_exactly_frozen():
    params = {
        "encoder": {"weight": jnp.ones((2,))},
        "value_head": {"weight": jnp.ones((2,))},
        "privileged_encoder": {"weight": jnp.ones((2,))},
        "critic_macro_encoder": {"weight": jnp.ones((2,))},
    }
    state = create_train_state(_Model(), {"params": params}, learning_rate=0.1, max_grad_norm=1.0)
    grads = {name: {"weight": jnp.ones((2,))} for name in params}
    updated = state.apply_gradients(grads=grads)
    assert jnp.array_equal(updated.params["encoder"]["weight"], params["encoder"]["weight"])
    assert not jnp.array_equal(
        updated.params["value_head"]["weight"], params["value_head"]["weight"]
    )
    assert not jnp.array_equal(
        updated.params["privileged_encoder"]["weight"],
        params["privileged_encoder"]["weight"],
    )
    assert not jnp.array_equal(
        updated.params["critic_macro_encoder"]["weight"],
        params["critic_macro_encoder"]["weight"],
    )


def test_critic_macro_features_are_player_relative():
    from kaggriculture.policy.jax.value_features import critic_macro_features
    from kaggriculture.simulator.reset import reset

    state = reset(jax.random.key(9), 1)
    state = jax.tree.map(lambda value: value[0], state)
    state = state._replace(
        money=state.money.at[0].set(9000).at[1].set(3000),
        hands_active=state.hands_active.at[0, :2].set(True),
    )

    player0 = critic_macro_features(state, jnp.asarray(0))
    player1 = critic_macro_features(state, jnp.asarray(1))

    np.testing.assert_allclose(player0.reshape(4, 3)[:, :2], player1.reshape(4, 3)[:, :2][:, ::-1])
    np.testing.assert_allclose(player0.reshape(4, 3)[:, 2], -player1.reshape(4, 3)[:, 2])


def test_completed_episode_targets_are_discounted_from_terminal_step(tmp_path, monkeypatch):
    import json
    from typing import NamedTuple

    import numpy as np

    from kaggriculture.training.value_pretrain import dataset

    class MockState(NamedTuple):
        step: np.ndarray

    monkeypatch.setattr(
        dataset,
        "paired_observations_to_state",
        lambda observations, **_: MockState(np.asarray(0)),
    )
    steps = [
        [{"observation": {"player": player}, "status": "ACTIVE"} for player in range(2)]
        for _ in range(2)
    ]
    steps.append([{"observation": {"player": player}, "status": "DONE"} for player in range(2)])
    path = tmp_path / "episode.json"
    path.write_text(json.dumps({"steps": steps, "rewards": [5, 3]}), encoding="utf-8")
    _, targets = dataset.load_episode(path, gamma=0.5, episode_steps=3, turns_per_day=24)
    np.testing.assert_allclose(targets, [[0.5, -0.5], [1.0, -1.0]])


def test_returns_combine_daily_asset_change_and_terminal_outcome(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from kaggriculture.training.value_pretrain import dataset

    states = SimpleNamespace(step=np.asarray([0, 24, 25]))
    assets = np.asarray([[0, 0], [100, 0], [200, 0]], dtype=np.float32)
    monkeypatch.setattr(dataset, "estimated_assets", lambda _: assets)
    targets = dataset._returns(
        states,
        1.0,
        gamma=0.5,
        turns_per_day=24,
        daily_reward_coefficient=1.0,
        daily_reward_scale=10000.0,
        daily_reward_maximum=1.0,
    )
    np.testing.assert_allclose(targets, [[0.515, -0.515], [1.01, -1.01]])


def _fake_episode(length: int, seed: int):
    from kaggriculture.simulator.reset import reset

    states = reset(jax.random.key(seed), length)
    # rng_keyはjaxのPRNGKey型でnp.asarrayできないため、実パイプライン
    # (paired_observations_to_state)と同じplainなuint32配列に置き換えてから
    # 残りのフィールドだけnumpyへ変換する。
    states = states._replace(rng_key=np.zeros((length, 2), dtype=np.uint32))
    states = states._replace(
        **{
            name: np.asarray(value)
            for name, value in zip(states._fields, states, strict=True)
            if name != "rng_key"
        }
    )
    targets = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
    return states, targets


def test_prepare_episode_reuses_an_existing_shard(tmp_path, monkeypatch):
    from kaggriculture.training.value_pretrain import cache

    calls = []

    def fake_load_episode(path, **kwargs):
        calls.append(path)
        return _fake_episode(3, seed=0)

    monkeypatch.setattr(cache, "load_episode", fake_load_episode)
    source = tmp_path / "episode.json"
    source.write_text("{}", encoding="utf-8")
    directory = tmp_path / "shards"
    kwargs = dict(gamma=0.999, episode_steps=4, turns_per_day=24)

    first = cache.prepare_episode(source, directory, **kwargs)
    second = cache.prepare_episode(source, directory, **kwargs)

    assert first is not None
    assert first == second
    assert len(calls) == 1  # 2回目はshardが既にあるので再変換しない

    states, targets = cache.load_shard(first)
    expected_states, expected_targets = _fake_episode(3, seed=0)
    np.testing.assert_allclose(targets, expected_targets)
    np.testing.assert_allclose(states.money, expected_states.money)


def test_prepare_episode_skips_incomplete_replays(tmp_path, monkeypatch):
    from kaggriculture.training.value_pretrain import cache

    def fake_load_episode(path, **kwargs):
        raise ValueError("incomplete episode")

    monkeypatch.setattr(cache, "load_episode", fake_load_episode)
    source = tmp_path / "episode.json"
    source.write_text("{}", encoding="utf-8")
    directory = tmp_path / "shards"

    result = cache.prepare_episode(
        source, directory, gamma=0.999, episode_steps=4, turns_per_day=24
    )

    assert result is None
    assert not directory.exists() or not list(directory.glob("*.npz"))


def test_iter_batches_spans_multiple_shards(tmp_path, monkeypatch):
    from kaggriculture.training.value_pretrain import cache

    episodes = {"a": _fake_episode(3, seed=1), "b": _fake_episode(2, seed=2)}

    def fake_load_episode(path, **kwargs):
        return episodes[path.stem]

    monkeypatch.setattr(cache, "load_episode", fake_load_episode)
    directory = tmp_path / "shards"
    sources = []
    for name in ("a", "b"):
        source = tmp_path / f"{name}.json"
        source.write_text("{}", encoding="utf-8")
        sources.append(source)
    shard_paths = cache.prepare_episodes(
        sources, directory, gamma=0.999, episode_steps=4, turns_per_day=24
    )

    batches = list(
        cache.iter_batches(shard_paths, batch_size=2, seed=0, shuffle=False, drop_last=False)
    )

    total = sum(len(targets) for _, targets in batches)
    assert total == 5  # 3件+2件、全て取りこぼさない
    assert all(len(targets) <= 2 for _, targets in batches)
