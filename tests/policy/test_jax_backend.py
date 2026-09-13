"""PyTorch提出モデルとJAX学習モデルの互換性テスト。"""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import actions as JA
from kaggriculture.policy.jax import distribution as JD
from kaggriculture.policy.jax import features as F
from kaggriculture.policy.jax import model as JM
from kaggriculture.policy.jax import tokenize as JT
from kaggriculture.policy.torch import distribution as TD
from kaggriculture.policy.torch import features as TF
from kaggriculture.policy.torch import model as TM
from kaggriculture.policy.torch import tokenize
from kaggriculture.rules import constants as C
from kaggriculture.simulator import market
from kaggriculture.training.weight_bridge import jax_to_torch, torch_to_jax
from scripts.trace_codec import build_state
from tests.policy.conftest import make_fresh_observation, make_fresh_private


def _vector(index: int, second: int | None = None) -> V.SparseVector:
    vector = V.SparseVector()
    vector.add(index % V.VOCAB_SIZE)
    if second is not None:
        vector.add(second % V.VOCAB_SIZE, 0.25)
    return vector


def _models():
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
    torch_net = TM.PolicyValueNet(config).eval()
    return torch_net, JM.PolicyValueNet(config), config


def test_weight_bridge_round_trip_is_exact() -> None:
    """PyTorch→JAX→PyTorchで全学習パラメータを保存する。"""
    torch_net, jax_net, config = _models()
    restored, _, _ = _models()
    variables = torch_to_jax(torch_net, config)
    jax_to_torch(variables, restored)
    for (name, expected), (restored_name, actual) in zip(
        torch_net.named_parameters(), restored.named_parameters(), strict=True
    ):
        assert name == restored_name
        assert torch.equal(expected, actual), name


def test_encoder_decoder_and_value_match() -> None:
    """同じ重みと固定入力で両backendの主要出力を一致させる。"""
    torch_net, jax_net, config = _models()
    variables = torch_to_jax(torch_net, config)

    encoder_tokens = [_vector(i, i + 7) for i in range(L.NUM_WORDS_ENCODER)]
    index, value, offset = TF.collate(encoder_tokens)
    with torch.no_grad():
        torch_memory = torch_net.encode(index, value, offset)
    packed = F.pack_batch([encoder_tokens], F.MAX_ENCODER_FEATURES)
    jax_memory = jax_net.apply(
        variables,
        jnp.asarray(packed.index),
        jnp.asarray(packed.value),
        method=jax_net.encode,
    )
    np.testing.assert_allclose(torch_memory.numpy(), np.asarray(jax_memory), atol=1e-3)

    decoder_tokens = [_vector(100 + i) for i in range(4)]
    d_index, d_value, d_offset = TF.collate(decoder_tokens)
    position_ids = torch.tensor([[0, 2, 4, 66]])
    board_position_ids = torch.tensor([[L.NO_POSITION, 1, 2, L.NO_POSITION]])
    padding_mask = torch.zeros((1, 4), dtype=torch.bool)
    with torch.no_grad():
        torch_hidden = torch_net.decoder(
            torch_memory,
            d_index,
            d_value,
            d_offset,
            position_ids,
            board_position_ids,
            padding_mask,
        )
        torch_value = torch_net.value(torch_memory)
    packed_decoder = F.pack_batch([decoder_tokens], F.MAX_DECODER_FEATURES)
    jax_hidden = jax_net.apply(
        variables,
        jax_memory,
        jnp.asarray(packed_decoder.index),
        jnp.asarray(packed_decoder.value),
        jnp.asarray(position_ids.numpy()),
        jnp.asarray(board_position_ids.numpy()),
        jnp.asarray(padding_mask.numpy()),
        method=jax_net.decode,
    )
    jax_value = jax_net.apply(variables, jax_memory, method=jax_net.get_value)
    np.testing.assert_allclose(torch_hidden.numpy(), np.asarray(jax_hidden), atol=1e-3)
    np.testing.assert_allclose(torch_value.squeeze(-1).numpy(), np.asarray(jax_value), atol=2e-4)


def _aggregate_features(index: np.ndarray, value: np.ndarray) -> np.ndarray:
    result = np.zeros((index.shape[0], V.VOCAB_SIZE), dtype=np.float32)
    rows = np.broadcast_to(np.arange(index.shape[0])[:, None], index.shape)
    np.add.at(result, (rows, index), value)
    return result


def test_jax_state_tokenization_matches_python_contract() -> None:
    """JAX State直結経路が既存Pythonトークンと同じ特徴和を作る。"""
    obs0 = make_fresh_observation(0)
    obs1 = make_fresh_observation(1)
    inventory = jnp.asarray([obs0["market"]["inventory"][item] for item in C.PRODUCTS])
    prices = np.asarray(market.market_price(inventory))
    obs0["market"]["prices"] = dict(zip(C.PRODUCTS, prices, strict=True))
    obs1["market"]["prices"] = dict(zip(C.PRODUCTS, prices, strict=True))
    state = build_state(obs0, obs1, V.BOARD_SIZE)

    expected = F.pack_vectors(tokenize.get_encoder_input(obs0), F.MAX_ENCODER_FEATURES)
    actual = JT.encode_observation(state, jnp.asarray(0))
    np.testing.assert_allclose(
        _aggregate_features(expected.index, expected.value),
        _aggregate_features(np.asarray(actual.index), np.asarray(actual.value)),
        atol=1e-6,
    )

    expected_tokens, expected_positions, expected_padding = tokenize.get_privileged_critic_input(
        obs0, make_fresh_private()
    )
    expected_privileged = F.pack_vectors(expected_tokens, F.MAX_PRIVILEGED_FEATURES)
    actual_privileged, actual_positions, actual_padding = JT.encode_privileged(
        state, jnp.asarray(0)
    )
    np.testing.assert_allclose(
        _aggregate_features(expected_privileged.index, expected_privileged.value),
        _aggregate_features(
            np.asarray(actual_privileged.index), np.asarray(actual_privileged.value)
        ),
        atol=1e-6,
    )
    np.testing.assert_array_equal(actual_positions, expected_positions)
    np.testing.assert_array_equal(actual_padding, expected_padding)


def test_teacher_forced_action_probability_matches_between_backends() -> None:
    """候補順が異なっても同じ複合行動の確率を一致させる。"""
    torch_net, jax_net, config = _models()
    variables = torch_to_jax(torch_net, config)
    obs0 = make_fresh_observation(0)
    obs1 = make_fresh_observation(1)
    inventory = jnp.asarray([obs0["market"]["inventory"][item] for item in C.PRODUCTS])
    prices = np.asarray(market.market_price(inventory))
    obs0["market"]["prices"] = dict(zip(C.PRODUCTS, prices, strict=True))
    obs1["market"]["prices"] = dict(zip(C.PRODUCTS, prices, strict=True))
    state = build_state(obs0, obs1, V.BOARD_SIZE)
    batched_state = jax.tree.map(lambda value: value[None], state)

    action = {"farmer": ["PASS"], "hands": [], "market": []}
    torch_log_prob, _, torch_decisions = TD.evaluate_policy(torch_net, obs0, action)
    pass_choice = int(
        np.flatnonzero(
            (np.asarray(JA.UNIT_CANDIDATES.op) == C.FARMER_OP_PASS)
            & (np.asarray(JA.UNIT_CANDIDATES.arg) < 0)
        )[0]
    )
    stop_choice = int(np.flatnonzero(np.asarray(JA.MARKET_CANDIDATES.op) == JA.MARKET_STOP)[0])
    choices = jnp.zeros((1, L.MAX_DECODE_LEN), dtype=jnp.int32)
    choices = choices.at[0, 0].set(pass_choice).at[0, 1].set(stop_choice)
    evaluated = JD.evaluate_choices(jax_net, variables, batched_state, jnp.asarray([0]), choices)

    assert torch_decisions == 2
    assert int(evaluated.num_decisions[0]) == 2
    np.testing.assert_allclose(
        np.asarray(evaluated.log_prob[0]), torch_log_prob.detach().numpy(), atol=2e-3
    )
