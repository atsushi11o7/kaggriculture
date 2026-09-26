"""Shop-demand features used by the state-driven opening."""

from __future__ import annotations

from kaggriculture.rules import constants as C
from kaggriculture.rules import game_params as P


def product_demand(unlocked_shops: list[str]) -> dict[str, int]:
    """Return additive product demand, preserving duplicate shop instances."""
    result = dict.fromkeys(C.PRODUCTS, 0)
    for shop in unlocked_shops:
        if shop not in C.SHOPS:
            continue
        row = P.SHOP_DEMAND[C.SHOPS.index(shop)]
        for product, amount in zip(C.PRODUCTS, row, strict=True):
            result[product] += amount
    return result
