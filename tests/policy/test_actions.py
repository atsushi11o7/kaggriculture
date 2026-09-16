"""PyTorch方策の軽量候補判定テスト。"""

from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.torch import actions as A
from kaggriculture.policy.torch import decode as DS
from kaggriculture.rules import constants as C


def test_has_executable_quantity_matches_maximum() -> None:
    """軽量な存在判定と完全な最大数量計算の境界を一致させる。"""
    market = {"inventory": dict.fromkeys(C.PRODUCTS, 400)}
    cases = (
        ("PICKUP", "WHEAT", None),
        ("PLACE", "WHEAT", {"WHEAT": 3}),
        ("BUY_SEED", C.CROPS[0], None),
        ("BUY_ANIMAL", C.ANIMALS[0], None),
        ("BUY_PRODUCT", "WHEAT", None),
        ("SELL", "WHEAT", None),
    )
    for money in (0, 1, 10, 100, 1_000, 100_000):
        for shed_count in (0, 1, 99, 100):
            farm = {"money": money}
            shed = dict.fromkeys(C.SHED_ITEMS, 0)
            shed["WHEAT"] = shed_count
            for op_name, item_name, inventory in cases:
                expected = (
                    A.max_executable_quantity(
                        op_name,
                        item_name,
                        farm,
                        shed,
                        market,
                        shed_capacity=100,
                        inventory=inventory,
                    )
                    > 0
                )
                actual = A.has_executable_quantity(
                    op_name,
                    item_name,
                    farm,
                    shed,
                    market,
                    shed_capacity=100,
                    inventory=inventory,
                )
                assert actual == expected, (op_name, money, shed_count)


def test_commit_unit_action_lets_a_later_unit_pickup_what_an_earlier_unit_placed() -> None:
    """先行unitが納屋へ置いた品目を後続unitが拾えることを確認する。"""
    farm = {"tiles": [[None] * 10 for _ in range(10)]}
    shed = dict.fromkeys(C.SHED_ITEMS, 0)
    seeds = dict.fromkeys(C.CROPS, 0)
    position = (4, 4)  # 納屋隣接マス
    farmer_inventory = {"FERTILIZER": 1}
    hand_inventory: dict = {}

    pickup_fertilizer = A._candidate(farmer_op="PICKUP", item_index=V.entity_index("FERTILIZER"))
    pickup_key = tuple(pickup_fertilizer.index)

    legal_before = {
        tuple(vector.index)
        for vector in A.legal_unit_actions(farm, shed, seeds, hand_inventory, position, 0, 100)
    }
    assert pickup_key not in legal_before  # ターン開始時点ではshedが空

    DS.commit_unit_action(
        farm,
        shed,
        seeds,
        "PLACE",
        "FERTILIZER",
        position,
        0,
        n=1,
        shed_capacity=100,
        inventory=farmer_inventory,
    )
    assert shed["FERTILIZER"] == 1

    legal_after = {
        tuple(vector.index)
        for vector in A.legal_unit_actions(farm, shed, seeds, hand_inventory, position, 0, 100)
    }
    assert pickup_key in legal_after  # 先行unitのPLACE後は後続unitのPICKUPが合法化する


def test_legal_unit_actions_accepts_observation_position_lists() -> None:
    """JSON由来のlist座標でも納屋隣接行動を生成する。"""
    farm = {"tiles": [[None] * 10 for _ in range(10)]}
    shed = dict.fromkeys(C.SHED_ITEMS, 0)
    shed["WHEAT"] = 1
    seeds = dict.fromkeys(C.CROPS, 0)
    candidates = A.legal_unit_actions(farm, shed, seeds, {}, [4, 4], 0)
    pickup = A._candidate(farmer_op="PICKUP", item_index=V.entity_index("WHEAT"))
    assert tuple(pickup.index) in {tuple(candidate.index) for candidate in candidates}
