"""Cross-backend contracts for deterministic endgame overrides."""

import jax
import jax.numpy as jnp

from kaggriculture.policy.endgame import jax as jax_endgame
from kaggriculture.policy.endgame import torch as torch_endgame
from kaggriculture.rules import constants as C
from kaggriculture.simulator.action import Action
from kaggriculture.simulator.reset import reset
from tests.policy.conftest import make_fresh_observation


def _blank_action() -> Action:
    return Action(
        jnp.asarray(C.FARMER_OP_PASS),
        jnp.asarray(0),
        jnp.asarray(1),
        jnp.full((C.MAX_HANDS,), C.FARMER_OP_PASS),
        jnp.zeros((C.MAX_HANDS,), jnp.int32),
        jnp.ones((C.MAX_HANDS,), jnp.int32),
        jnp.full((C.MAX_MARKET_ORDERS,), -1),
        jnp.zeros((C.MAX_MARKET_ORDERS,), jnp.int32),
        jnp.zeros((C.MAX_MARKET_ORDERS,), jnp.int32),
    )


def test_jax_final_step_drops_cargo_and_liquidates_available_products() -> None:
    state = jax.tree.map(lambda value: value[0], reset(jax.random.key(0), 1))
    state = state._replace(
        step=jnp.asarray(718),
        farmer_pos=state.farmer_pos.at[0].set(jnp.asarray([4, 4])),
        farmer_inventory=state.farmer_inventory.at[0, 0].set(3),
        shed=state.shed.at[0, 1].set(2),
    )

    action, unit_override, market_override = jax_endgame.apply(
        state, jnp.asarray(0), _blank_action()
    )

    assert int(action.farmer_op) == C.FARMER_OP_DROP
    assert bool(unit_override[0])
    assert action.market_op[:2].tolist() == [C.MARKET_OP_SELL, C.MARKET_OP_SELL]
    assert action.market_arg_idx[:2].tolist() == [0, 1]
    assert action.market_n[:2].tolist() == [3, 2]
    assert bool(market_override.all())


def test_torch_final_step_matches_drop_and_liquidation_contract() -> None:
    obs = make_fresh_observation(day=29)
    obs["hour"] = 22
    obs["farms"][0]["farmer"] = [4, 4]
    obs["private"]["inventories"][0]["WHEAT"] = 3
    obs["private"]["shed"]["CARROT"] = 2

    action = torch_endgame.apply(
        obs,
        {"farmer": ["PASS"], "hands": [], "market": []},
    )

    assert action["farmer"] == ["DROP"]
    assert action["market"] == [["SELL", "WHEAT", 3], ["SELL", "CARROT", 2]]


def test_return_starts_at_last_feasible_step() -> None:
    obs = make_fresh_observation(day=29)
    obs["hour"] = 14  # step 710: nine actions remain through step 718.
    obs["farms"][0]["farmer"] = [0, 0]
    obs["private"]["inventories"][0]["WHEAT"] = 1

    action = torch_endgame.apply(
        obs,
        {"farmer": ["PASS"], "hands": [], "market": []},
    )

    assert action["farmer"] == ["EAST"]
