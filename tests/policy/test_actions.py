"""PyTorch方策の軽量候補判定テスト。"""

from kaggriculture.policy.torch import actions as A
from kaggriculture.simulator import constants as C


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
