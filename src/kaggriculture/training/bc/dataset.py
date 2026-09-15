"""正規化済みリプレイを非自己回帰方策の固定slot教師へ変換する。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from kaggriculture.policy.jax import actions as A
from kaggriculture.policy.jax.types import Intent
from kaggriculture.rules import constants as C
from kaggriculture.simulator.state import State
from kaggriculture.training.replays.io import iter_replay_samples
from kaggriculture.training.replays.state import CacheRules, observation_to_state


class BCBatch(NamedTuple):
    states: State
    players: np.ndarray
    intent: Intent
    slot_mask: np.ndarray


@dataclass
class BuildStats:
    accepted: int = 0
    discarded: int = 0
    reasons: Counter[str] = field(default_factory=Counter)


def _lookup(table) -> dict[tuple[int, int], int]:
    return {
        (int(op), int(arg)): index
        for index, (op, arg) in enumerate(zip(table.op, table.arg, strict=True))
    }


_UNIT = _lookup(A.UNIT_CANDIDATES)
_MARKET = _lookup(A.MARKET_CANDIDATES)
_PASS = _UNIT[(C.FARMER_OP_PASS, -1)]
_STOP = _MARKET[(A.MARKET_STOP, -1)]
_WAIT = _MARKET[(A.MARKET_WAIT, -1)]


def _unit(entry: list) -> tuple[int, int]:
    op = C.FARMER_OP_NAMES.index(entry[0])
    if op == C.FARMER_OP_PLANT:
        arg = C.CROPS.index(entry[1])
    elif op in (C.FARMER_OP_PICKUP, C.FARMER_OP_PLACE):
        arg = C.SHED_ITEMS.index(entry[1])
    else:
        arg = -1
    quantity = int(entry[2]) if len(entry) > 2 else 1
    if not 1 <= quantity <= 100:
        raise ValueError("invalid_quantity")
    return _UNIT[(op, arg)], quantity - 1


def _market(entry: list) -> tuple[int, int]:
    if entry == ["SELL", "WHEAT", 0]:
        return _WAIT, 0
    op = C.MARKET_OP_NAMES.index(entry[0])
    if op == C.MARKET_OP_BUY_SEED:
        arg = C.CROPS.index(entry[1])
    elif op == C.MARKET_OP_BUY_ANIMAL:
        arg = C.ANIMALS.index(entry[1])
    elif op in (C.MARKET_OP_BUY_PRODUCT, C.MARKET_OP_SELL):
        arg = C.PRODUCTS.index(entry[1])
    else:
        arg = -1
    quantity = int(entry[2]) if len(entry) > 2 else 1
    if not 1 <= quantity <= 100:
        raise ValueError("invalid_quantity")
    return _MARKET[(op, arg)], quantity - 1


def action_to_intent(obs: dict, action: dict, rules: CacheRules):
    """expertを正規化し、全unit・市場10slotの教師indexへ変換する。"""
    from kaggriculture.policy.torch.candidate_api import normalize_expert_action

    normalized = normalize_expert_action(
        obs,
        action,
        turns_per_day=rules.turns_per_day,
        shed_capacity=rules.shed_capacity,
        hire_mult=rules.hire_mult,
        max_market_orders=rules.max_market_orders,
    )
    hands = obs["farms"][obs["player"]]["hands"]
    if len(hands) > C.MAX_HANDS:
        raise ValueError("too_many_hands")
    unit = np.full(C.MAX_HANDS + 1, _PASS, np.int32)
    unit_quantity = np.zeros(C.MAX_HANDS + 1, np.int32)
    unit_mask = np.zeros(C.MAX_HANDS + 1, bool)
    unit_mask[: len(hands) + 1] = True
    for index, entry in enumerate([normalized["farmer"], *normalized["hands"]]):
        unit[index], unit_quantity[index] = _unit(entry)
    market = np.full(C.MAX_MARKET_ORDERS, _STOP, np.int32)
    market_quantity = np.zeros(C.MAX_MARKET_ORDERS, np.int32)
    market_mask = np.zeros(C.MAX_MARKET_ORDERS, bool)
    for index, entry in enumerate(normalized["market"][: C.MAX_MARKET_ORDERS]):
        market[index], market_quantity[index] = _market(entry)
        market_mask[index] = True
    if len(normalized["market"]) < C.MAX_MARKET_ORDERS:
        market_mask[len(normalized["market"])] = True
    return Intent(unit, unit_quantity, market, market_quantity), np.concatenate(
        [unit_mask, market_mask]
    )


def iter_samples(sources, rules: CacheRules, stats: BuildStats):
    """不正サンプルを理由別に集計して逐次変換する。"""
    for item in sources:
        path, selected_players = item if isinstance(item, tuple) else (item, None)
        for obs, action in iter_replay_samples(
            Path(path), rules.min_player_reward, set(selected_players) if selected_players else None
        ):
            try:
                intent, mask = action_to_intent(obs, action, rules)
                state = observation_to_state(obs, turns_per_day=rules.turns_per_day)
                stats.accepted += 1
                yield state, int(obs["player"]), intent, mask
            except (KeyError, IndexError, TypeError, ValueError) as error:
                stats.discarded += 1
                stats.reasons[type(error).__name__ + ":" + str(error)[:80]] += 1


def stack_samples(samples) -> BCBatch:
    samples = list(samples)
    if not samples:
        raise ValueError("empty BC batch")
    states = State(
        *(
            np.stack([np.asarray(getattr(sample[0], name)) for sample in samples])
            for name in State._fields
        )
    )
    intent = Intent(
        *(
            np.stack([np.asarray(getattr(sample[2], name)) for sample in samples])
            for name in Intent._fields
        )
    )
    return BCBatch(
        states,
        np.asarray([sample[1] for sample in samples], np.int32),
        intent,
        np.stack([sample[3] for sample in samples]),
    )
