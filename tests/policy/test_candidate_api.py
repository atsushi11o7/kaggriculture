"""外部候補採点器向け公開decode APIの契約テスト。"""

import pytest

from kaggriculture.policy.torch.candidate_api import (
    candidate_action_names,
    decode_with_candidates,
    trace_expert_candidates,
)
from tests.policy.conftest import make_fresh_observation


def test_decode_callback_receives_all_decisions_and_returns_action() -> None:
    observation = make_fresh_observation(0)
    kinds = []

    def choose(decision):
        kinds.append(decision.kind)
        return len(decision.candidates) - 1 if decision.kind == "market_op" else 0

    action = decode_with_candidates(observation, choose)

    assert kinds == ["unit_op", "market_op"]
    assert set(action) == {"farmer", "hands", "market"}
    assert action["hands"] == []
    assert action["market"] == []


def test_decode_rejects_candidate_index_outside_public_contract() -> None:
    with pytest.raises(ValueError, match="outside"):
        decode_with_candidates(make_fresh_observation(0), lambda decision: -1)


def test_expert_trace_exposes_selected_candidate_before_commit() -> None:
    observation = make_fresh_observation(0)
    seen = []
    action = {"farmer": ["PASS"], "hands": [], "market": []}

    trace_expert_candidates(
        observation, action, lambda decision, selected: seen.append((decision, selected))
    )

    assert [decision.kind for decision, _ in seen] == ["unit_op", "market_op"]
    assert all(0 <= selected < len(decision.candidates) for decision, selected in seen)
    assert seen[0][0].context.farm["money"] == 3000


def test_candidate_names_hide_sparse_vocabulary_details() -> None:
    observation = make_fresh_observation(0)
    seen = []

    def choose(decision):
        seen.append(candidate_action_names(decision.candidates[0], decision.kind))
        return len(decision.candidates) - 1 if decision.kind == "market_op" else 0

    decode_with_candidates(observation, choose)

    assert seen == [("PASS", None), ("HIRE", None)]
