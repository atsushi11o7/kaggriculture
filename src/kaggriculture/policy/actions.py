"""今合法な行動候補だけを列挙し、SparseVectorにエンコードする。

方策ネットは固定サイズの行動空間全体にスコアを付けるのではなく、この関数群が
列挙した候補だけをデコーダへの入力として受け取る。存在しない(=合法でない)
候補にはそもそもスコアという概念が無いため、合法性は後付けのマスクではなく
構造として保証される。

合法条件はsimulator側の各applies条件(crop_actions.py/animal_actions.py/
inventory_actions.py/market_orders.py/market_lockstep.py)と対応させている。
ただしこちらは生のkaggle-environments observation(dict)を直接読む、
学習・推論共通のPython実装。
"""

from kaggriculture.policy import vocab as V
from kaggriculture.simulator import constants as C
from kaggriculture.simulator import game_params as P

_FARMER_OP = {name: i for i, name in enumerate(C.FARMER_OP_NAMES)}
_MARKET_OP = {name: i for i, name in enumerate(C.MARKET_OP_NAMES)}
_MOVE_DELTA = dict(
    zip(("NORTH", "SOUTH", "EAST", "WEST"), ((0, -1), (0, 1), (1, 0), (-1, 0)), strict=True)
)
_ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}


def _is_shed_adjacent(pos: tuple[int, int], board_size: int) -> bool:
    half = board_size // 2
    return pos in ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def _candidate(farmer_op=None, market_op=None, item_index=None) -> V.SparseVector:
    sv = V.SparseVector()
    if farmer_op is not None:
        sv.add(V.ACTION_FARMER_OP[_FARMER_OP[farmer_op]])
    if market_op is not None:
        sv.add(V.ACTION_MARKET_OP[_MARKET_OP[market_op]])
    if item_index is not None:
        sv.add(item_index)
    return sv


def legal_unit_actions(
    farm: dict, shed: dict, seeds: dict, inventory: dict, pos: tuple[int, int]
) -> list:
    """1ユニット(farmerまたは1体のhand)の、今合法な行動候補を列挙する。

    Args:
        farm: obs["farms"][player](このプレイヤーの畑)。
        shed: このプレイヤーの納屋(private["shed"])。PICKUP判定に使う。
        seeds: このプレイヤーの残り種(private["seeds"])。PLANT判定に使う。
        inventory: このユニット自身の持ち物(private["inventories"][idx])。
        pos: (x, y) このユニットの現在位置。

    Returns:
        list[SparseVector]: 合法な(op, item)候補のリスト。PASS(常に合法)を含む。
    """
    board_size = len(farm["tiles"])
    fx, fy = pos
    tile = farm["tiles"][fy][fx]
    shed_adjacent = _is_shed_adjacent(pos, board_size)

    candidates = [_candidate(farmer_op="PASS")]

    for name, (dx, dy) in _MOVE_DELTA.items():
        nx, ny = fx + dx, fy + dy
        if 0 <= nx < board_size and 0 <= ny < board_size:
            candidates.append(_candidate(farmer_op=name))

    is_plant = isinstance(tile, dict) and tile.get("kind") == "PLANT"
    is_weed = isinstance(tile, dict) and tile.get("kind") == "WEED"
    is_structure = isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE")
    has_animal = is_structure and tile.get("animal") is not None
    is_empty = tile is None

    if is_plant:
        if not tile["watered_today"]:
            candidates.append(_candidate(farmer_op="WATER"))
        if tile["yield_units"] > 0:
            candidates.append(_candidate(farmer_op="HARVEST"))
        if inventory.get("FERTILIZER", 0) > 0:
            candidates.append(_candidate(farmer_op="FERTILIZE"))
        candidates.append(_candidate(farmer_op="DIG"))

    if has_animal:
        if tile["yield_units"] > 0:
            candidates.append(_candidate(farmer_op="HARVEST"))
        if not tile["fed_today"] and inventory.get("WHEAT", 0) > 0:
            candidates.append(_candidate(farmer_op="FEED"))
        if not tile["cared_today"]:
            candidates.append(_candidate(farmer_op="CARE"))
        if tile["fertilizer_available"]:
            candidates.append(_candidate(farmer_op="COLLECT_FERTILIZER"))

    if is_weed:
        candidates.append(_candidate(farmer_op="DIG"))

    # 空の小屋に立っている間、その小屋に対応する動物名のPLACEは(持ち物の有無に
    # 関わらず)常に動物配置側の判定を専有し、納屋落としにはフォールバックしない
    # (animal_actions.pyのon_animal_branch参照)。この動物名だけ、下のPLACE
    # (納屋落とし)候補の対象から除外する。
    on_animal_branch_item = None
    if is_structure and not has_animal:
        candidates.append(_candidate(farmer_op="DIG"))
        on_animal_branch_item = next(a for a, s in _ANIMAL_STRUCTURE.items() if s == tile["kind"])
        if inventory.get(on_animal_branch_item, 0) > 0:
            candidates.append(
                _candidate(
                    farmer_op="PLACE",
                    item_index=V.TILE_ANIMAL[C.ANIMALS.index(on_animal_branch_item)],
                )
            )

    if is_empty:
        for crop, n in seeds.items():
            if n > 0:
                candidates.append(
                    _candidate(farmer_op="PLANT", item_index=V.TILE_CROP[C.CROPS.index(crop)])
                )
        candidates.append(_candidate(farmer_op="BUILD_COOP"))
        candidates.append(_candidate(farmer_op="BUILD_PASTURE"))

    if shed_adjacent:
        if any(n > 0 for n in inventory.values()):
            candidates.append(_candidate(farmer_op="DROP"))
            for item, n in inventory.items():
                if n > 0 and item != on_animal_branch_item:
                    candidates.append(
                        _candidate(
                            farmer_op="PLACE", item_index=V.SHED_ITEM[C.SHED_ITEMS.index(item)]
                        )
                    )
        for item, n in shed.items():
            if n > 0:
                candidates.append(
                    _candidate(farmer_op="PICKUP", item_index=V.SHED_ITEM[C.SHED_ITEMS.index(item)])
                )

    return candidates


def legal_market_actions(
    farm: dict,
    shed: dict,
    market: dict,
    hire_mult: float = 1,
    shed_capacity: int = 100,
) -> list:
    """市場注文キューの1スロット分として、今合法な行動候補を列挙する。

    Args:
        farm: obs["farms"][player]。money, unlocked_quadrants, hires_todayを使う。
        shed: このプレイヤーの納屋(private["shed"])。SELL/BUY_ANIMAL等の容量判定に使う。
        market: obs["market"]。inventory/pricesを使う。
        hire_mult: farmHandCostMult。
        shed_capacity: 納屋の容量。

    Returns:
        list[SparseVector]: 合法な(op, item)候補のリスト。
    """
    money = farm["money"]
    shed_total = sum(shed.values())
    candidates = []

    hire_cost = _fib(farm["hires_today"]) * hire_mult
    if money >= hire_cost:
        candidates.append(_candidate(market_op="HIRE"))

    n_unlocked_extra = len(farm["unlocked_quadrants"]) - 1
    if n_unlocked_extra < len(P.LAND_PRICES) and money >= P.LAND_PRICES[n_unlocked_extra]:
        candidates.append(_candidate(market_op="BUY_LAND"))

    for i in range(C.N_CROPS):
        if money >= P.CROP_SEED_COST[i]:
            candidates.append(_candidate(market_op="BUY_SEED", item_index=V.TILE_CROP[i]))

    if shed_total < shed_capacity:
        for i in range(C.N_ANIMALS):
            if money >= P.ANIMAL_COST[i]:
                candidates.append(_candidate(market_op="BUY_ANIMAL", item_index=V.TILE_ANIMAL[i]))
        for item in ("WHEAT", "FERTILIZER"):
            if money >= market["prices"][item]:
                candidates.append(
                    _candidate(
                        market_op="BUY_PRODUCT", item_index=V.MARKET_PRODUCT[C.PRODUCTS.index(item)]
                    )
                )

    for item, n in shed.items():
        if n > 0 and item in C.PRODUCTS:
            candidates.append(
                _candidate(market_op="SELL", item_index=V.MARKET_PRODUCT[C.PRODUCTS.index(item)])
            )

    return candidates


def _fib(n: int) -> int:
    """fib(0)=1, fib(1)=1, fib(2)=2, ... (HIREのn人目のコスト倍率)。"""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a
