"""Fixed action vocabularies shared by Torch inference and execution."""

from kaggriculture.policy.common import action_schema as S
from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.torch import actions
from kaggriculture.rules import constants as C


def _candidate(op: int, arg: int, *, market: bool) -> V.SparseVector:
    vector = V.SparseVector()
    table = V.ACTION_MARKET_OP if market else V.ACTION_FARMER_OP
    vector.add(table.start + op)
    if arg >= 0:
        entity = C.N_PRODUCTS + arg if market and op == C.MARKET_OP_BUY_ANIMAL else arg
        vector.add(V.ENTITY_ITEM.start + entity)
    return vector


def _vectors(metadata: tuple[tuple[int, int], ...], *, market: bool) -> list[V.SparseVector]:
    vectors = []
    for op, arg in metadata:
        if market and op == S.MARKET_WAIT:
            vectors.append(actions.market_wait_candidate())
        elif market and op == S.MARKET_STOP:
            vectors.append(actions.market_stop_candidate())
        else:
            vectors.append(_candidate(op, arg, market=market))
    return vectors


UNIT_META = list(S.UNIT_CANDIDATES)
MARKET_META = list(S.MARKET_CANDIDATES)
UNIT_VECTORS = _vectors(S.UNIT_CANDIDATES, market=False)
MARKET_VECTORS = _vectors(S.MARKET_CANDIDATES, market=True)
QUANTITY_VECTORS = actions.quantity_candidates(V.MAX_ACTION_QUANTITY)


def key(vector: V.SparseVector) -> tuple[int, ...]:
    """Return the stable identity used to compare sparse candidates."""
    return tuple(vector.index)


def item_name(op: int, arg: int, *, market: bool) -> str | None:
    """Convert fixed candidate metadata to the simulator item name."""
    if arg < 0:
        return None
    if market and op == C.MARKET_OP_BUY_ANIMAL:
        return C.ANIMALS[arg]
    if (market and op == C.MARKET_OP_BUY_SEED) or (not market and op == C.FARMER_OP_PLANT):
        return C.CROPS[arg]
    return C.SHED_ITEMS[arg]
