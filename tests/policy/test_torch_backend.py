"""Cross-backend checks for the fixed-slot policy."""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import features as F
from kaggriculture.policy.jax.model import N_UNIT_SLOTS
from kaggriculture.policy.jax.model import PolicyValueNet as JaxNet
from kaggriculture.policy.jax.policy import initialize
from kaggriculture.policy.torch.model import PolicyValueNet as TorchNet
from kaggriculture.policy.torch.policy import predict_action
from kaggriculture.training.weight_bridge import jax_to_torch
from tests.policy.conftest import make_fresh_observation


def test_jax_to_torch_outputs_match() -> None:
    config = ModelConfig(
        d_model=8,
        num_heads=1,
        d_feedforward=16,
        num_layers_encoder=1,
        num_layers_decoder=1,
        num_layers_critic=1,
        dropout=0.0,
        use_episode_history=False,
        use_asymmetric_critic=False,
    )
    jax_net = JaxNet(config)
    variables = initialize(jax_net, jax.random.key(0), batch_size=2)
    torch_net = TorchNet(config).eval()
    jax_to_torch(variables, torch_net)

    index = np.zeros((2, L.NUM_WORDS_ENCODER, F.MAX_ENCODER_FEATURES), np.int32)
    value = np.zeros_like(index, dtype=np.float32)
    index[:, :, 0] = np.arange(L.NUM_WORDS_ENCODER)[None] % 10
    value[:, :, 0] = 1.0
    positions = np.full((2, N_UNIT_SLOTS), L.NO_POSITION, np.int32)
    positions[:, 0] = 0
    active = np.zeros((2, N_UNIT_SLOTS), bool)
    active[:, 0] = True
    inventory_index = np.zeros((2, N_UNIT_SLOTS, F.MAX_ENCODER_FEATURES), np.int32)
    inventory_value = np.zeros_like(inventory_index, dtype=np.float32)
    inventory_index[:, 0, 0] = 5
    inventory_value[:, 0, 0] = 1.0

    jq, jv = jax_net.apply(
        variables,
        jnp.asarray(index),
        jnp.asarray(value),
        jnp.asarray(positions),
        jnp.asarray(active),
        jnp.asarray(inventory_index),
        jnp.asarray(inventory_value),
    )
    with torch.no_grad():
        tq, tv = torch_net(
            torch.from_numpy(index),
            torch.from_numpy(value),
            torch.from_numpy(positions),
            torch.from_numpy(active),
            torch.from_numpy(inventory_index),
            torch.from_numpy(inventory_value),
        )
    # XLA and PyTorch attention reductions differ slightly over the long state sequence.
    np.testing.assert_allclose(np.asarray(jq), tq.numpy(), atol=3e-3, rtol=3e-3)
    np.testing.assert_allclose(np.asarray(jv), tv.numpy(), atol=3e-3, rtol=3e-3)


def test_torch_predicts_complete_action() -> None:
    config = ModelConfig(
        d_model=8,
        num_heads=1,
        d_feedforward=16,
        num_layers_encoder=1,
        num_layers_decoder=1,
        num_layers_critic=1,
        dropout=0.0,
        use_episode_history=False,
        use_asymmetric_critic=False,
    )
    action = predict_action(TorchNet(config), make_fresh_observation())
    assert set(action) == {"farmer", "hands", "market"}
    assert action["farmer"]
    assert action["hands"] == []
    assert len(action["market"]) <= 10
