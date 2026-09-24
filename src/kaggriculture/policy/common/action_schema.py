"""Backend-independent ordering and sizes of the fixed action vocabulary."""

from kaggriculture.policy.common import vocab as V
from kaggriculture.rules import constants as C

MARKET_WAIT = C.N_MARKET_OPS
MARKET_STOP = C.N_MARKET_OPS + 1

_ITEM_UNIT_OPS = {C.FARMER_OP_PICKUP, C.FARMER_OP_PLANT, C.FARMER_OP_PLACE}
UNIT_CANDIDATES = tuple((op, -1) for op in range(C.N_FARMER_OPS) if op not in _ITEM_UNIT_OPS)
UNIT_CANDIDATES += tuple((C.FARMER_OP_PLANT, item) for item in range(C.N_CROPS))
UNIT_CANDIDATES += tuple((C.FARMER_OP_PLACE, item) for item in range(C.N_SHED_ITEMS))
UNIT_CANDIDATES += tuple((C.FARMER_OP_PICKUP, item) for item in range(C.N_SHED_ITEMS))

MARKET_CANDIDATES = ((C.MARKET_OP_HIRE, -1), (C.MARKET_OP_BUY_LAND, -1))
MARKET_CANDIDATES += tuple((C.MARKET_OP_BUY_SEED, item) for item in range(C.N_CROPS))
MARKET_CANDIDATES += tuple((C.MARKET_OP_BUY_ANIMAL, item) for item in range(C.N_ANIMALS))
MARKET_CANDIDATES += tuple(
    (C.MARKET_OP_BUY_PRODUCT, C.PRODUCTS.index(item)) for item in ("WHEAT", "FERTILIZER")
)
MARKET_CANDIDATES += tuple((C.MARKET_OP_SELL, item) for item in range(C.N_PRODUCTS))
MARKET_CANDIDATES += ((MARKET_WAIT, -1), (MARKET_STOP, -1))

N_UNIT_ACTIONS = len(UNIT_CANDIDATES)
N_MARKET_ACTIONS = len(MARKET_CANDIDATES)
N_QUANTITIES = V.MAX_ACTION_QUANTITY
