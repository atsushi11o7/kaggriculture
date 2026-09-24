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
from kaggriculture.policy.torch import history as torch_history
from kaggriculture.policy.torch import policy as torch_policy
from kaggriculture.policy.torch.model import PolicyValueNet as TorchNet
from kaggriculture.policy.torch.policy import predict_action
from kaggriculture.training.weight_bridge import jax_to_torch
from tests.policy.conftest import make_fresh_observation


def test_jax_to_torch_asymmetric_critic_outputs_match() -> None:
    config = ModelConfig(
        d_model=8,
        num_heads=1,
        d_feedforward=16,
        num_layers_encoder=1,
        num_layers_decoder=1,
        num_layers_critic=1,
    )
    jax_net = JaxNet(config)
    variables = initialize(jax_net, jax.random.key(10), batch_size=2)
    torch_net = TorchNet(config).eval()
    jax_to_torch(variables, torch_net)

    index = np.zeros((2, L.NUM_WORDS_ENCODER, F.MAX_ENCODER_FEATURES), np.int32)
    value = np.zeros_like(index, dtype=np.float32)
    positions = np.full((2, N_UNIT_SLOTS), L.NO_POSITION, np.int32)
    active = np.zeros((2, N_UNIT_SLOTS), bool)
    active[:, 0] = True
    inventory_index = np.zeros((2, N_UNIT_SLOTS, F.MAX_ENCODER_FEATURES), np.int32)
    inventory_value = np.zeros_like(inventory_index, dtype=np.float32)

    count = len(L.PRIVILEGED_OWNER_ZONE_WITH_CLS) - 1
    privileged_index = np.zeros((2, count, F.MAX_PRIVILEGED_FEATURES), np.int32)
    privileged_value = np.zeros_like(privileged_index, dtype=np.float32)
    privileged_positions = np.full((2, count), L.NO_POSITION, np.int32)
    privileged_padding = np.zeros((2, count), bool)
    critic_macro = np.linspace(-0.5, 0.5, 24, dtype=np.float32).reshape(2, 12)

    args = (
        index,
        value,
        positions,
        active,
        inventory_index,
        inventory_value,
        privileged_index,
        privileged_value,
        privileged_positions,
        privileged_padding,
        critic_macro,
    )
    jq, jv = jax_net.apply(variables, *(jnp.asarray(item) for item in args))
    with torch.no_grad():
        tq, tv = torch_net(*(torch.from_numpy(item) for item in args))

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
    )
    action = predict_action(
        TorchNet(config, actor_only=True), make_fresh_observation(), counters={}
    )
    assert set(action) == {"farmer", "hands", "market"}
    assert action["farmer"]
    assert action["hands"] == []
    assert len(action["market"]) <= 10


def test_torch_place_quantity_bound_uses_unit_inventory() -> None:
    """TorchとJAXがPLACE数量を同じ手持ち数で制約する。"""
    farm = {"tiles": [[None] * 10 for _ in range(10)]}
    op = next(
        op
        for op, arg in torch_policy.UNIT_META
        if op == torch_policy.C.FARMER_OP_PLACE and arg == 0
    )
    bound = torch_policy._unit_quantity_upper_bound(
        op, 0, (4, 4), {torch_policy.C.SHED_ITEMS[0]: 3}, farm
    )
    assert bound == 3


def test_torch_executor_skips_stale_duplicate_harvest_commit() -> None:
    """同じ一発収穫型作物への後続HARVESTでshadow更新を繰り返さない。"""
    from kaggriculture.rules import constants as C
    from kaggriculture.rules import game_params as P

    crop_index = next(index for index, ongoing in enumerate(P.CROP_IS_ONGOING) if not ongoing)
    crop = C.CROPS[crop_index]
    day = P.CROP_FIRST_YIELD_DAY[crop_index]
    farm = {"tiles": [[None] * 10 for _ in range(10)]}
    farm["tiles"][0][0] = {
        "kind": "PLANT",
        "crop": crop,
        "watered_today": True,
        "consecutive_unwatered": 0,
        "fertilized_until_day": -1,
        "yield_units": 1,
        "planted_day": 0,
        "max_lifespan_step": 10_000,
    }
    shed = dict.fromkeys(C.SHED_ITEMS, 0)
    seeds = dict.fromkeys(C.CROPS, 0)
    harvest = next(
        index for index, (op, _) in enumerate(torch_policy.UNIT_META) if op == C.FARMER_OP_HARVEST
    )

    entries = torch_policy._execute_units(
        [harvest, harvest],
        [1, 1],
        [(0, 0), (0, 0)],
        [{}, {}],
        farm,
        shed,
        seeds,
        day,
        turns_per_day=24,
        shed_capacity=100,
    )

    assert entries == [["HARVEST"], ["HARVEST"]]
    assert farm["tiles"][0][0] is None


def test_torch_history_replays_duplicate_harvest_as_one_production() -> None:
    """History replay ignores a HARVEST invalidated by the preceding unit."""
    from kaggriculture.rules import constants as C
    from kaggriculture.rules import game_params as P

    crop_index = next(index for index, ongoing in enumerate(P.CROP_IS_ONGOING) if not ongoing)
    crop = C.CROPS[crop_index]
    day = P.CROP_FIRST_YIELD_DAY[crop_index]
    farm = {
        "farmer": (0, 0),
        "hands": [(0, 0)],
        "tiles": [[None] * 10 for _ in range(10)],
    }
    farm["tiles"][0][0] = {
        "kind": "PLANT",
        "crop": crop,
        "watered_today": True,
        "consecutive_unwatered": 0,
        "fertilized_until_day": -1,
        "yield_units": 1,
        "planted_day": 0,
        "max_lifespan_step": 10_000,
    }
    shed = dict.fromkeys(C.SHED_ITEMS, 0)
    seeds = dict.fromkeys(C.CROPS, 0)
    market = {
        "inventory": dict.fromkeys(C.PRODUCTS, 400),
        "prices": dict.fromkeys(C.PRODUCTS, 1.0),
    }
    action = {
        "farmer": ["HARVEST"],
        "hands": [["HARVEST"]],
        "market": [],
    }

    deltas = torch_history.compute_turn_deltas(farm, shed, seeds, market, [{}, {}], day, action)

    assert deltas == {"produced": {crop: 1}}
