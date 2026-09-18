"""異なるvariables同士の対戦・座席バイアス除去の契約テスト。"""

import jax
import pytest

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import model as M
from kaggriculture.policy.jax import policy as P
from kaggriculture.training.ppo.evaluation import (
    EvaluationResult,
    _combine_seats,
    evaluate_both_seats,
    evaluate_closed_loop,
)
from kaggriculture.training.ppo.rollout import RolloutConfig


def _model():
    config = ModelConfig(8, 1, 16, 1, 1, 0.0, False, False, 1)
    model = M.PolicyValueNet(config)
    return model, P.initialize(model, jax.random.key(0))


def test_evaluate_closed_loop_runs_to_terminal_state() -> None:
    model, variables = _model()
    config = RolloutConfig(episode_steps=8)

    result = evaluate_closed_loop(model, variables, variables, config, jax.random.key(1), 4)

    assert result.cash.shape == (4, 2)
    assert result.outcome.shape == (4,)
    assert 0.0 <= float(result.win_rate) <= 1.0


def test_combine_seats_flips_and_negates_the_second_assignment() -> None:
    seat0 = EvaluationResult(
        cash=jax.numpy.asarray([[10.0, 2.0]]),
        outcome=jax.numpy.asarray([1.0]),
        win_rate=jax.numpy.asarray(1.0),
        pass_rate=jax.numpy.asarray([0.1]),
        opponent_pass_rate=jax.numpy.asarray([0.2]),
    )
    seat1 = EvaluationResult(
        cash=jax.numpy.asarray([[3.0, 9.0]]),
        outcome=jax.numpy.asarray([1.0]),  # seat1視点でopponent(候補側player1)が勝った
        win_rate=jax.numpy.asarray(1.0),
        pass_rate=jax.numpy.asarray([0.3]),
        opponent_pass_rate=jax.numpy.asarray([0.4]),
    )

    combined = _combine_seats(seat0, seat1)

    # seat1側はcashの列を反転(常に候補視点=列0)し、outcomeの符号を反転する。
    assert combined.cash.tolist() == [[10.0, 2.0], [9.0, 3.0]]
    assert combined.outcome.tolist() == [1.0, -1.0]
    assert float(combined.win_rate) == 0.5
    # pass_rateもcashと同様、候補視点(seat1側はopponent_pass_rateが候補)に揃える。
    assert combined.pass_rate.tolist() == pytest.approx([0.1, 0.4])
    assert combined.opponent_pass_rate.tolist() == pytest.approx([0.2, 0.3])


def test_combine_seats_gives_half_credit_for_an_exact_tie() -> None:
    # greedy同士の対称な対局は完全な引き分け(outcome==0)になりやすい。
    # mean(outcome > 0)だけだと引き分けを負け扱いしてしまうので、0.5点であること
    # を固定する回帰テスト。
    tie = EvaluationResult(
        cash=jax.numpy.asarray([[5.0, 5.0]]),
        outcome=jax.numpy.asarray([0.0]),
        win_rate=jax.numpy.asarray(0.0),
        pass_rate=jax.numpy.asarray([0.0]),
        opponent_pass_rate=jax.numpy.asarray([0.0]),
    )

    combined = _combine_seats(tie, tie)

    assert float(combined.win_rate) == 0.5


def test_evaluate_both_seats_is_close_to_half_against_itself() -> None:
    model, variables = _model()
    config = RolloutConfig(episode_steps=8, temperature=1.0)

    result = evaluate_both_seats(model, variables, variables, config, jax.random.key(3), 8)

    assert result.cash.shape == (16, 2)
    assert 0.0 <= float(result.win_rate) <= 1.0
