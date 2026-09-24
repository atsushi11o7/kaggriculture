"""非自己回帰JAX方策の固定shape・再評価契約。"""

import jax
import jax.numpy as jnp

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import actions as A
from kaggriculture.policy.jax import executor as E
from kaggriculture.policy.jax import model as M
from kaggriculture.policy.jax import policy as P
from kaggriculture.policy.jax.types import Intent
from kaggriculture.rules import constants as C
from kaggriculture.simulator.action import Action
from kaggriculture.simulator.reset import reset
from kaggriculture.simulator.step import step_batch_lockstep


def _model():
    config = ModelConfig(16, 2, 32, 1, 1, 1)
    model = M.PolicyValueNet(config)
    return model, P.initialize(model, jax.random.PRNGKey(0), batch_size=2)


def test_policy_samples_and_re_evaluates() -> None:
    model, variables = _model()
    states = reset(jax.random.PRNGKey(1), 2)
    players = jnp.asarray([0, 1], dtype=jnp.int32)
    output = P.sample_actions(model, variables, states, players, jax.random.PRNGKey(2), greedy=True)
    evaluated = P.evaluate_intent(model, variables, states, players, output.intent)

    assert output.intent.unit.shape == (2, C.MAX_HANDS + 1)
    assert output.intent.market.shape == (2, C.MAX_MARKET_ORDERS)
    assert output.action.hands_op.shape == (2, C.MAX_HANDS)
    assert jnp.allclose(evaluated.log_prob, output.log_prob, atol=1e-5)
    assert jnp.allclose(evaluated.slot_log_prob, output.slot_log_prob, atol=1e-5)
    assert jnp.allclose(evaluated.value, output.value, atol=1e-5)


def test_unit_inventory_changes_the_encoded_query() -> None:
    """own_inventoryは全unit合計なので、query側で各unit固有のinventoryを
    別途注入していないと、farmer/hand毎にPLACE/DROP/SELL等の対象unitを
    区別できない(レビューで指摘・修正した問題の回帰テスト)。"""
    model, variables = _model()
    base = reset(jax.random.PRNGKey(5), 1)  # batch=1、hands_active等は(1, 2, ...)
    player = 0
    hand_slot = 2  # unit index = hand_slot + 1 (0はfarmer)
    active = base.hands_active.at[0, player, hand_slot].set(True)
    base = base._replace(hands_active=active)
    wheat = C.SHED_ITEMS.index("WHEAT")
    with_item = base.hands_inventory.at[0, player, hand_slot, wheat].set(3)
    empty = base
    with_item_state = base._replace(hands_inventory=with_item)
    players = jnp.asarray([player], jnp.int32)

    def _queries(state):
        _, units, market, *_ = P._logits(model, variables, state, players, None, 24, 100)
        return jnp.concatenate([units, market], axis=1)

    queries_empty = _queries(empty)
    queries_with_item = _queries(with_item_state)
    assert not jnp.allclose(queries_empty, queries_with_item)


def test_executor_resolves_same_turn_place_then_pickup() -> None:
    """Executorが同turnのPLACE後のPICKUPを納屋在庫へ反映する。"""
    state = reset(jax.random.PRNGKey(0), 1)
    player = 0
    fertilizer = C.SHED_ITEMS.index("FERTILIZER")

    hands_active = state.hands_active.at[0, player, 0].set(True)
    # 納屋隣接マス(default_spawn_position、farmerと同じ)へhand0を配置する。
    hands_pos = state.hands_pos.at[0, player, 0].set(state.farmer_pos[0, player])
    farmer_inventory = state.farmer_inventory.at[0, player, fertilizer].set(1)
    state = state._replace(
        hands_active=hands_active, hands_pos=hands_pos, farmer_inventory=farmer_inventory
    )
    assert int(state.shed[0, player, fertilizer]) == 0  # ターン開始時点では在庫0

    place_index = int(
        jnp.nonzero(
            (A.UNIT_CANDIDATES.op == C.FARMER_OP_PLACE) & (A.UNIT_CANDIDATES.arg == fertilizer)
        )[0][0]
    )
    pickup_index = int(
        jnp.nonzero(
            (A.UNIT_CANDIDATES.op == C.FARMER_OP_PICKUP) & (A.UNIT_CANDIDATES.arg == fertilizer)
        )[0][0]
    )
    pass_index = int(jnp.nonzero(A.UNIT_CANDIDATES.op == C.FARMER_OP_PASS)[0][0])
    stop_index = int(jnp.nonzero(A.MARKET_CANDIDATES.op == A.MARKET_STOP)[0][0])

    unit = jnp.full((1, C.MAX_HANDS + 1), pass_index, jnp.int32)
    unit = unit.at[0, 0].set(place_index)  # farmer: PLACE(FERTILIZER)
    unit = unit.at[0, 1].set(pickup_index)  # hand0: PICKUP(FERTILIZER)
    unit_quantity = jnp.zeros((1, C.MAX_HANDS + 1), jnp.int32)  # 量index0 = 実量1個
    market = jnp.full((1, C.MAX_MARKET_ORDERS), stop_index, jnp.int32)
    market_quantity = jnp.zeros((1, C.MAX_MARKET_ORDERS), jnp.int32)
    intent0 = Intent(unit, unit_quantity, market, market_quantity)
    # player 1側はPASS/STOPのみの何もしないintent(実在するので合法・無関係)。
    intent1 = intent0._replace(unit=jnp.full_like(unit, pass_index))

    action0, stats0 = E.execute(state, jnp.asarray([0], jnp.int32), intent0)
    action1, _ = E.execute(state, jnp.asarray([1], jnp.int32), intent1)
    action = Action(*(jnp.stack([a, b], axis=1) for a, b in zip(action0, action1, strict=True)))

    assert int(stats0.invalid_unit[0]) == 0
    assert int(action0.farmer_op[0]) == C.FARMER_OP_PLACE
    assert int(action0.hands_op[0, 0]) == C.FARMER_OP_PICKUP

    next_state, _, _ = step_batch_lockstep(state, action)
    assert int(next_state.hands_inventory[0, player, 0, fertilizer]) == 1
    assert int(next_state.farmer_inventory[0, player, fertilizer]) == 0
    assert int(next_state.shed[0, player, fertilizer]) == 0


def test_self_play_action_steps_simulator() -> None:
    model, variables = _model()
    states = reset(jax.random.PRNGKey(3), 2)
    output = jax.jit(P.sample_self_play_actions, static_argnums=(0,))(
        model, variables, states, jax.random.PRNGKey(4), greedy=True
    )
    next_state, _, _ = step_batch_lockstep(states, output.action)

    assert output.action.farmer_op.shape == (2, 2)
    assert output.intent.market.shape == (2, 2, C.MAX_MARKET_ORDERS)
    assert output.value.shape == (2, 2)
    assert next_state.step.tolist() == [1, 1]
