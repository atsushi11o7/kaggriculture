"""strategy maskのバックエンド一致と採点時の適用契約。"""

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture.policy.common.strategy import StrategyMaskConfig, market_candidate_allowed
from kaggriculture.policy.jax import actions as A
from kaggriculture.policy.jax import policy as P
from kaggriculture.policy.jax import strategy as JS
from kaggriculture.policy.torch import policy as TP
from kaggriculture.policy.torch import strategy as TS
from kaggriculture.rules import constants as C
from kaggriculture.simulator.reset import reset
from tests.policy.conftest import make_fresh_observation
from tests.policy.test_policy import _model


def test_strategy_masks_match_between_backends() -> None:
    state = reset(jax.random.key(0), 1)
    obs = make_fresh_observation()
    jax_market = np.asarray(JS.market_mask(state, jnp.asarray(0)))
    torch_market = np.asarray(TS.market_mask(obs, TP.MARKET_META))
    np.testing.assert_array_equal(jax_market, torch_market)

    wait = next(i for i, (op, _) in enumerate(TP.MARKET_META) if op == C.N_MARKET_OPS)
    stop = next(i for i, (op, _) in enumerate(TP.MARKET_META) if op == C.N_MARKET_OPS + 1)
    assert jax_market[:-1, wait].all()
    assert not jax_market[-1, wait]
    assert jax_market[:, stop].all()
    assert jax_market.sum(axis=-1).min() > 0

    jax_unit = np.asarray(JS.unit_mask(state, jnp.asarray(0), jnp.asarray(0)))
    torch_unit = np.asarray(TS.unit_mask(obs, 0, TP.UNIT_META))
    np.testing.assert_array_equal(jax_unit, torch_unit)
    assert jax_unit.all()

    assert market_candidate_allowed(
        C.MAX_MARKET_ORDERS - 1,
        A.MARKET_WAIT,
        max_slots=C.MAX_MARKET_ORDERS,
        wait_op=A.MARKET_WAIT,
        config=StrategyMaskConfig(exclude_final_market_wait=False),
    )


def test_strategy_mask_is_shared_by_sampling_and_re_evaluation() -> None:
    model, variables = _model()
    states = reset(jax.random.key(1), 2)
    players = jnp.asarray([0, 1], dtype=jnp.int32)
    output = P.sample_actions(model, variables, states, players, jax.random.key(2))
    evaluated = P.evaluate_intent(model, variables, states, players, output.intent)
    np.testing.assert_allclose(
        np.asarray(output.slot_log_prob), np.asarray(evaluated.slot_log_prob)
    )
    wait = int(jnp.nonzero(A.MARKET_CANDIDATES.op == A.MARKET_WAIT)[0][0])
    assert not np.asarray(output.intent.market[:, -1] == wait).any()
