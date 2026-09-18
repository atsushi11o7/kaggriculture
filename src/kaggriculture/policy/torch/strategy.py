"""Torch提出推論へ適用する戦略上の候補制約。"""

from kaggriculture.policy.common.strategy import market_candidate_allowed
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
    del obs, unit
    return [True] * len(candidates)


def market_mask(obs: dict, candidates: list[tuple[int, int]]) -> list[list[bool]]:
    """市場slotと候補の戦略maskを返す。

    Args:
        obs: 提出推論で受け取る観測。
        candidates: 候補の操作IDと引数。

    Returns:
        市場slotと候補の許可mask。
    """
    del obs
    return [
        [
            market_candidate_allowed(
                slot, op, max_slots=C.MAX_MARKET_ORDERS, wait_op=C.N_MARKET_OPS
            )
            for op, _ in candidates
        ]
        for slot in range(C.MAX_MARKET_ORDERS)
    ]
