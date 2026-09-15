"""合法候補をLightGBM向け固定幅の数値特徴へ変換する。"""

from __future__ import annotations

import numpy as np

from kaggriculture.policy.common import vocab as V
from kaggriculture.rules import constants as C

FEATURE_VERSION = 3
_SLOT_KINDS = ("unit_op", "market_op", "quantity")
_TILE_KINDS = ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE")

CONTEXT_NAMES = (
    "day",
    "hour",
    "slot_position",
    "unit_x",
    "unit_y",
    "own_money",
    "opponent_money",
    "hands",
    "hires_today",
    "shed_total",
    "tile_age",
    "tile_yield",
    "tile_uncared",
    "tile_watered_or_fed",
    "tile_cared",
    "tile_fertilized",
    "tile_fertilizer_available",
    "remaining_days",
    *(f"slot_{name}" for name in _SLOT_KINDS),
    *(f"tile_{name}" for name in _TILE_KINDS),
    *(f"crop_{name}" for name in C.CROPS),
    *(f"animal_{name}" for name in C.ANIMALS),
    *(f"shed_{name}" for name in C.SHED_ITEMS),
    *(f"seed_{name}" for name in C.CROPS),
    *(f"price_{name}" for name in C.PRODUCTS),
    *(f"market_inventory_{name}" for name in C.PRODUCTS),
)
CANDIDATE_NAMES = (
    *(f"farmer_op_{name}" for name in C.FARMER_OP_NAMES),
    *(f"market_op_{name}" for name in C.MARKET_OP_NAMES),
    *(f"item_{name}" for name in C.SHED_ITEMS),
    "market_wait",
    "market_stop",
    "quantity",
    "quantity_log",
)
RESULT_NAMES = (
    "target_x",
    "target_y",
    "target_yield",
    "target_uncared",
    "candidate_shed_count",
    "candidate_seed_count",
    "candidate_market_inventory",
    "candidate_market_price",
    *(f"target_tile_{name}" for name in _TILE_KINDS),
    *(f"target_crop_{name}" for name in C.CROPS),
    *(f"target_animal_{name}" for name in C.ANIMALS),
)
FEATURE_NAMES = CONTEXT_NAMES + CANDIDATE_NAMES + RESULT_NAMES

_MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}

# market_opの行1本から、どのop名(STOP/HIRE/BUY_SEED等)の候補かを復元するための
# 逆引き表。候補IDそのものは特徴量へ残していないが、one-hotフラグから一意に
# 復元できるので、RankingData(features/labels)だけでクラス別の評価ができる。
MARKET_OP_CLASS_NAMES = (*C.MARKET_OP_NAMES, "WAIT", "STOP")
_MARKET_OP_FLAG_INDICES = tuple(
    FEATURE_NAMES.index(f"market_op_{name}") for name in C.MARKET_OP_NAMES
)
_MARKET_WAIT_INDEX = FEATURE_NAMES.index("market_wait")
_MARKET_STOP_INDEX = FEATURE_NAMES.index("market_stop")


def classify_market_op_row(row: np.ndarray) -> str:
    """market_op特徴行1本から、対応するop名(MARKET_OP_CLASS_NAMESの要素)を返す。"""
    if row[_MARKET_WAIT_INDEX]:
        return "WAIT"
    if row[_MARKET_STOP_INDEX]:
        return "STOP"
    for name, index in zip(C.MARKET_OP_NAMES, _MARKET_OP_FLAG_INDICES, strict=True):
        if row[index]:
            return name
    raise ValueError("row does not encode a recognizable market_op candidate")


def context_features(
    obs: dict, env, slot_kind: str, position: int, board_position: int, total_days: int = 30
):
    """その決定時点の公開情報と自分のshadow stateを数値化する。"""
    player = int(obs["player"])
    farm = env.farm
    opponent = obs["farms"][1 - player]
    x = board_position % V.BOARD_SIZE if board_position < V.BOARD_SIZE**2 else -1
    y = board_position // V.BOARD_SIZE if x >= 0 else -1
    tile = farm["tiles"][y][x] if x >= 0 else None
    kind = "EMPTY" if tile is None else "LOCKED" if tile == "LOCKED" else tile.get("kind", "EMPTY")
    age = (
        obs["day"] - tile.get("planted_day", tile.get("placed_day", obs["day"]))
        if isinstance(tile, dict)
        else 0
    )
    values = [
        obs["day"],
        obs["hour"],
        position,
        x,
        y,
        farm["money"],
        opponent["money"],
        len(farm["hands"]),
        farm["hires_today"],
        sum(env.shed.values()),
        age,
        tile.get("yield_units", 0) if isinstance(tile, dict) else 0,
        tile.get("consecutive_unwatered", tile.get("consecutive_unfed", 0))
        if isinstance(tile, dict)
        else 0,
        float(tile.get("watered_today", tile.get("fed_today", False)))
        if isinstance(tile, dict)
        else 0,
        float(tile.get("cared_today", False)) if isinstance(tile, dict) else 0,
        float(tile.get("fertilized_until_day", -1) >= obs["day"]) if isinstance(tile, dict) else 0,
        float(tile.get("fertilizer_available", False)) if isinstance(tile, dict) else 0,
        max(total_days - obs["day"], 0),
    ]
    values.extend(float(slot_kind == name) for name in _SLOT_KINDS)
    values.extend(float(kind == name) for name in _TILE_KINDS)
    values.extend(float(isinstance(tile, dict) and tile.get("crop") == name) for name in C.CROPS)
    values.extend(
        float(isinstance(tile, dict) and tile.get("animal") == name) for name in C.ANIMALS
    )
    values.extend(env.shed.get(name, 0) for name in C.SHED_ITEMS)
    values.extend(env.seeds.get(name, 0) for name in C.CROPS)
    values.extend(env.market["prices"].get(name, 0) for name in C.PRODUCTS)
    values.extend(env.market["inventory"].get(name, 0) for name in C.PRODUCTS)
    return np.asarray(values, dtype=np.float32)


def _candidate_identity(candidate: V.SparseVector) -> tuple[str | None, str | None]:
    op = None
    item = None
    for index in candidate.index:
        if index in V.ACTION_FARMER_OP:
            op = C.FARMER_OP_NAMES[index - V.ACTION_FARMER_OP.start]
        elif index in V.ACTION_MARKET_OP:
            op = C.MARKET_OP_NAMES[index - V.ACTION_MARKET_OP.start]
        elif index in V.ENTITY_ITEM:
            item = C.SHED_ITEMS[index - V.ENTITY_ITEM.start]
    return op, item


def candidate_features(candidate: V.SparseVector, env, board_position: int) -> np.ndarray:
    """候補の意味と、移動先・対象資源を含む局所的な実行結果を数値化する。"""
    values = np.zeros(len(CANDIDATE_NAMES) + len(RESULT_NAMES), dtype=np.float32)
    for index in candidate.index:
        if index in V.ACTION_FARMER_OP:
            values[index - V.ACTION_FARMER_OP.start] = 1
        elif index in V.ACTION_MARKET_OP:
            offset = len(C.FARMER_OP_NAMES)
            values[offset + index - V.ACTION_MARKET_OP.start] = 1
        elif index in V.ENTITY_ITEM:
            offset = len(C.FARMER_OP_NAMES) + len(C.MARKET_OP_NAMES)
            values[offset + index - V.ENTITY_ITEM.start] = 1
        elif index in V.ACTION_MARKET_WAIT:
            values[len(CANDIDATE_NAMES) - 4] = 1
        elif index in V.ACTION_MARKET_STOP:
            values[len(CANDIDATE_NAMES) - 3] = 1
        elif index in V.ACTION_QUANTITY_VALUE:
            quantity = index - V.ACTION_QUANTITY_VALUE.start + 1
            values[len(CANDIDATE_NAMES) - 2] = quantity
            values[len(CANDIDATE_NAMES) - 1] = np.log1p(quantity)

    result = values[len(CANDIDATE_NAMES) :]
    op, item = _candidate_identity(candidate)
    x = board_position % V.BOARD_SIZE if board_position < V.BOARD_SIZE**2 else -1
    y = board_position // V.BOARD_SIZE if x >= 0 else -1
    if op in _MOVES and x >= 0:
        dx, dy = _MOVES[op]
        x, y = x + dx, y + dy
    result[0:2] = (x, y)
    tile = env.farm["tiles"][y][x] if x >= 0 else None
    kind = "EMPTY" if tile is None else "LOCKED" if tile == "LOCKED" else tile.get("kind", "EMPTY")
    if isinstance(tile, dict):
        result[2] = tile.get("yield_units", 0)
        result[3] = tile.get("consecutive_unwatered", tile.get("consecutive_unfed", 0))
    if item is not None:
        result[4] = env.shed.get(item, 0)
        result[5] = env.seeds.get(item, 0)
        result[6] = env.market["inventory"].get(item, 0)
        result[7] = env.market["prices"].get(item, 0)
    offset = 8
    for name in _TILE_KINDS:
        result[offset] = kind == name
        offset += 1
    for name in C.CROPS:
        result[offset] = isinstance(tile, dict) and tile.get("crop") == name
        offset += 1
    for name in C.ANIMALS:
        result[offset] = isinstance(tile, dict) and tile.get("animal") == name
        offset += 1
    return values


def group_features(
    context: np.ndarray, candidates: list[V.SparseVector], env, board_position: int
) -> np.ndarray:
    """同じ決定に属する全候補を一度に特徴量行列へする。"""
    repeated = np.broadcast_to(context, (len(candidates), len(context)))
    candidate_rows = np.stack([candidate_features(c, env, board_position) for c in candidates])
    return np.concatenate([repeated, candidate_rows], axis=1)
