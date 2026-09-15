"""Transformer以外の候補採点器向け、安定した自己回帰デコード契約。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from kaggriculture.policy.common.vocab import SparseVector
from kaggriculture.policy.torch import candidate_decode as D


@dataclass(frozen=True)
class DecodeContext:
    """現在の決定まで反映された自分の仮状態。読み取り専用として扱う。"""

    farm: dict
    shed: dict
    seeds: dict
    market: dict


@dataclass(frozen=True)
class CandidateDecision:
    """1つの自己回帰スロットの合法候補と特徴量用文脈。"""

    candidates: list[SparseVector]
    kind: str
    position: int
    board_position: int
    context: DecodeContext


def _run(observation: dict, choose: Callable[[CandidateDecision], int], rules: dict) -> dict:
    env = D._envs_from_observations([observation], **rules)[0]
    generator = D._decode_turn_gen(env)
    try:
        pending = next(generator)
        while True:
            candidates, kind, position, board_position, _ = pending
            decision = CandidateDecision(
                candidates,
                kind,
                position,
                board_position,
                DecodeContext(env.farm, env.shed, env.seeds, env.market),
            )
            selected = int(choose(decision))
            if not 0 <= selected < len(candidates):
                raise ValueError(f"selected candidate {selected} outside [0, {len(candidates)})")
            pending = generator.send(selected)
    except StopIteration as finished:
        return finished.value


def candidate_action_names(candidate: SparseVector, kind: str) -> tuple[str | None, str | None]:
    """候補のop名とitem名を、内部語彙表現に依存しない形で返す。"""
    if kind == "quantity":
        return None, None
    if kind == "unit_op":
        from kaggriculture.policy.common import vocab as V
        from kaggriculture.rules import constants as C

        op = D._op_name(candidate, V.ACTION_FARMER_OP, C.FARMER_OP_NAMES)
    elif kind == "market_op":
        from kaggriculture.policy.common import vocab as V
        from kaggriculture.rules import constants as C

        if V.ACTION_MARKET_WAIT.start in candidate.index:
            return "WAIT", None
        if V.ACTION_MARKET_STOP.start in candidate.index:
            return "STOP", None
        op = D._op_name(candidate, V.ACTION_MARKET_OP, C.MARKET_OP_NAMES)
    else:
        raise ValueError(f"unknown candidate kind: {kind}")
    return op, D._item_name(candidate)


def decode_with_candidates(
    observation: dict,
    choose: Callable[[CandidateDecision], int],
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1.0,
    max_market_orders: int = 10,
) -> dict:
    """各スロットで合法候補を採点・選択し、Kaggle形式の行動を返す。"""
    return _run(
        observation,
        choose,
        dict(
            turns_per_day=turns_per_day,
            shed_capacity=shed_capacity,
            hire_mult=hire_mult,
            max_market_orders=max_market_orders,
        ),
    )


def trace_expert_candidates(
    observation: dict,
    action: dict,
    visit: Callable[[CandidateDecision, int], None],
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1.0,
    max_market_orders: int = 10,
) -> None:
    """正規化したexpertの各候補表と正解indexを、状態更新前にvisitへ渡す。"""
    rules = dict(
        turns_per_day=turns_per_day,
        shed_capacity=shed_capacity,
        hire_mult=hire_mult,
        max_market_orders=max_market_orders,
    )
    normalized = D.normalize_expert_action(observation, action, **rules)
    chooser = D._TeacherForceChooser(normalized)

    def choose(decision: CandidateDecision) -> int:
        selected = chooser.choose(decision.candidates, decision.kind)
        visit(decision, selected)
        return selected

    _run(observation, choose, rules)


def normalize_expert_action(observation: dict, action: dict, **rules) -> dict:
    """Normalize replay actions against the same candidates used by tracing."""
    return D.normalize_expert_action(observation, action, **rules)
