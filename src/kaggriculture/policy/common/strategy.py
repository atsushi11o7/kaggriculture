"""モデル重みに依存しないstrategy maskの共有規則。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyMaskConfig:
    """採点前に適用する戦略上の候補制約。"""

    exclude_final_market_wait: bool = True


DEFAULT_STRATEGY_MASK = StrategyMaskConfig()


def market_candidate_allowed(
    slot: int,
    candidate_op: int,
    *,
    max_slots: int,
    wait_op: int,
    config: StrategyMaskConfig = DEFAULT_STRATEGY_MASK,
) -> bool:
    """最終市場slotで結果がSTOPと等しいWAITだけを除外する。

    Args:
        slot: 判定する市場slot。
        candidate_op: 候補の操作ID。
        max_slots: 市場slotの総数。
        wait_op: WAITの操作ID。
        config: 有効化する戦略規則。

    Returns:
        候補を残す場合はTrue。
    """
    return not (
        config.exclude_final_market_wait and slot == max_slots - 1 and candidate_op == wait_op
    )
