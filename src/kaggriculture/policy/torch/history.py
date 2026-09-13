"""確定済み行動から自分のエピソード累積実績を更新する。

produced/soldは確定値だが、市場の同時処理を単独再生できないためbought/revenueは
見積もりである。PPOバッファには各ターンで使用したcountersのコピーを保存すること。
"""

import copy

from kaggriculture.policy.torch import decode as DS

_CATEGORIES = ("produced", "sold", "estimated_bought_product", "estimated_revenue")


def compute_turn_deltas(
    farm: dict,
    shed: dict,
    seeds: dict,
    market: dict,
    inventories: list,
    day: int,
    action: dict,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1,
) -> dict[str, dict[str, int | float]]:
    """行動を内部コピー上で再生し、成立した量の差分を返す。

    Args:
        farm: ターン開始時の自農場。
        shed: ターン開始時の納屋。
        seeds: ターン開始時の種。
        market: ターン開始時の市場。
        inventories: farmer、hands順の持ち物。
        day: 現在日。
        action: 確定済み行動。
        turns_per_day: 1日当たりのターン数。
        shed_capacity: 納屋容量。
        hire_mult: 雇用費倍率。

    Returns:
        カテゴリ・品目ごとの成立量。
    """
    farm = copy.deepcopy(farm)
    shed = dict(shed)
    seeds = dict(seeds)
    market = copy.deepcopy(market)
    inventories = copy.deepcopy(inventories)

    deltas: dict[str, dict[str, int | float]] = {}

    def merge(delta: dict[str, dict[str, int | float]]) -> None:
        for category, per_item in delta.items():
            bucket = deltas.setdefault(category, {})
            for item, n in per_item.items():
                bucket[item] = bucket.get(item, 0) + n

    def do_unit(entry: list, pos: tuple[int, int], inventory: dict) -> None:
        op_name = entry[0]
        item_name = entry[1] if len(entry) > 1 else None
        n = entry[2] if len(entry) > 2 else 1
        merge(
            DS.commit_unit_action(
                farm,
                shed,
                seeds,
                op_name,
                item_name,
                pos,
                day,
                n=n,
                shed_capacity=shed_capacity,
                inventory=inventory,
                turns_per_day=turns_per_day,
            )
        )

    do_unit(action["farmer"], tuple(farm["farmer"]), inventories[0])
    for h, hand_pos in enumerate(farm["hands"]):
        do_unit(action["hands"][h], tuple(hand_pos), inventories[h + 1])

    for entry in action["market"]:
        op_name = entry[0]
        item_name = entry[1] if len(entry) > 1 else None
        n = entry[2] if len(entry) > 2 else 1
        merge(
            DS.commit_market_action(
                farm, shed, market, op_name, item_name, n, shed_capacity, hire_mult
            )
        )

    return deltas


def update_counters(counters: dict, deltas: dict) -> dict:
    """累積実績へ差分を加えた新しい辞書を返す。

    Args:
        counters: 更新前の累積実績。
        deltas: compute_turn_deltasの返り値。

    Returns:
        入力を変更せずに更新した累積実績。
    """
    new = {category: dict(counters.get(category, {})) for category in _CATEGORIES}
    for category in _CATEGORIES:
        for item, n in deltas.get(category, {}).items():
            new[category][item] = new[category].get(item, 0) + n

    has_ever_sold = dict(counters.get("has_ever_sold", {}))
    for item in deltas.get("sold", {}):
        has_ever_sold[item] = True
    new["has_ever_sold"] = has_ever_sold
    return new
