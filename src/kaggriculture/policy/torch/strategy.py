"""Torch提出推論へ適用する戦略上の候補制約。"""

from kaggriculture.policy.common.strategy import market_candidate_allowed
from kaggriculture.policy.endgame.common import investment_allowed
from kaggriculture.rules import constants as C


def unit_mask(obs: dict, unit: int, candidates: list[tuple[int, int]]) -> list[bool]:
    """unit候補の戦略maskを返す。

    Args:
        obs: 提出推論で受け取る観測。
        unit: 対象unit slot。
        candidates: 候補の操作IDと引数。

    Returns:
        候補順の許可mask。現時点では全候補を許可する。
    """
    del unit
    return [investment_allowed(op, arg, market=False, day=obs["day"]) for op, arg in candidates]


def market_mask(obs: dict, candidates: list[tuple[int, int]]) -> list[list[bool]]:
    """市場slotと候補の戦略maskを返す。

    Args:
        obs: 提出推論で受け取る観測。
        candidates: 候補の操作IDと引数。

    Returns:
        市場slotと候補の許可mask。
    """
    return [
        [
            market_candidate_allowed(
                slot, op, max_slots=C.MAX_MARKET_ORDERS, wait_op=C.N_MARKET_OPS
            )
            and investment_allowed(op, arg, market=True, day=obs["day"])
            for op, arg in candidates
        ]
        for slot in range(C.MAX_MARKET_ORDERS)
    ]
