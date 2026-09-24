"""合法な行動だけを列挙し、共有語彙のSparseVectorへ変換する。

合法条件はsimulatorのapply条件に合わせる。固定行動空間への後付けmaskではなく、
存在する候補だけを方策へ渡すPython実装である。
"""

import math

import numpy as np

from kaggriculture.policy.common import vocab as V
from kaggriculture.rules import constants as C
from kaggriculture.rules import game_params as P

_FARMER_OP = {name: i for i, name in enumerate(C.FARMER_OP_NAMES)}
_MARKET_OP = {name: i for i, name in enumerate(C.MARKET_OP_NAMES)}
_MOVE_DELTA = dict(
    zip(("NORTH", "SOUTH", "EAST", "WEST"), ((0, -1), (0, 1), (1, 0), (-1, 0)), strict=True)
)
_ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}

# Kaggle環境では数量0の注文は状態を変えないが、キューの1スロットを占める。
MARKET_WAIT_ACTION = ("SELL", "WHEAT", 0)


def _is_shed_adjacent(pos: tuple[int, int] | list[int], board_size: int) -> bool:
    half = board_size // 2
    x, y = pos
    return x in (half - 1, half) and y in (half - 1, half)


def is_animal_placement(item_name: str | None, tile) -> bool:
    """PLACE(item_name)候補が「動物配置」(空の対応する小屋に置く)を意味するか
    判定する。Falseなら「納屋落とし」(shed-drop)の意味になる。

    同じ(op="PLACE", item_name)の候補ベクトルが、今立っているタイルによって
    どちらの意味にもなりうる(動物を持っていても、対応する空の小屋の上に
    いなければ納屋落としにしかならない)ため、legal_unit_actions(候補生成)と
    decode_state.commit_unit_action/requires_quantity(数量の要否・状態更新)の
    両方でこの1関数を使い、判定基準がずれないようにする。
    """
    return (
        item_name in C.ANIMALS
        and isinstance(tile, dict)
        and tile.get("animal") is None
        and tile.get("kind") == _ANIMAL_STRUCTURE.get(item_name)
    )


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
    farm: dict,
    shed: dict,
    seeds: dict,
    inventory: dict,
    pos: tuple[int, int],
    day: int,
    shed_capacity: int = 100,
    *,
    defer_shared_resources: bool = False,
) -> list:
    """1ユニット(farmerまたは1体のhand)の、今合法な行動候補を列挙する。

    Args:
        farm: obs["farms"][player](このプレイヤーの畑)。
        shed: このプレイヤーの納屋(private["shed"])。PICKUP判定に使う。
        seeds: このプレイヤーの残り種(private["seeds"])。PLANT判定に使う。
        inventory: このユニット自身の持ち物(private["inventories"][idx])。
        pos: (x, y) このユニットの現在位置。
        day: 現在の日。作物HARVESTの成熟判定(下記参照)に使う。
        shed_capacity: 納屋の容量。PLACE(納屋落とし)の空き容量判定に使う。
        defer_shared_resources: Trueなら納屋在庫と空き容量の判定をExecutorへ委ねる。

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
        # 一発収穫型は植付直後からyield_units=1だが、first_yield_day経過するまでは
        # 収穫できない(crop_actions.apply_harvestのplant_matured参照)。
        first_yield_day = P.CROP_FIRST_YIELD_DAY[C.CROPS.index(tile["crop"])]
        matured = (day - tile["planted_day"]) >= first_yield_day
        if tile["yield_units"] > 0 and matured:
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

    # 空の小屋に立っている間、その小屋に対応する動物名(PASTUREはCOW・SHEEPの
    # 2種類が該当)のPLACEは常に動物配置側の意味になり、納屋落としにはならない
    # (animal_actions.pyのon_animal_branch参照)。これらの動物名は、下の
    # PLACE(納屋落とし)候補の対象から除外する。
    on_animal_branch_items = tuple(a for a in C.ANIMALS if is_animal_placement(a, tile))
    if is_structure and not has_animal:
        candidates.append(_candidate(farmer_op="DIG"))
    for animal in on_animal_branch_items:
        if inventory.get(animal, 0) > 0:
            candidates.append(_candidate(farmer_op="PLACE", item_index=V.entity_index(animal)))

    if is_empty:
        for crop in C.CROPS:
            if seeds.get(crop, 0) > 0:
                candidates.append(_candidate(farmer_op="PLANT", item_index=V.entity_index(crop)))
        candidates.append(_candidate(farmer_op="BUILD_COOP"))
        candidates.append(_candidate(farmer_op="BUILD_PASTURE"))

    if shed_adjacent:
        has_inventory = any(n > 0 for n in inventory.values())
        room = sum(shed.values()) < shed_capacity
        if has_inventory and (defer_shared_resources or room):
            candidates.append(_candidate(farmer_op="DROP"))
        if has_inventory:
            for item in C.SHED_ITEMS:
                if inventory.get(item, 0) <= 0 or item in on_animal_branch_items:
                    continue
                if defer_shared_resources or room:
                    candidates.append(
                        _candidate(farmer_op="PLACE", item_index=V.entity_index(item))
                    )
        for item in C.SHED_ITEMS:
            if defer_shared_resources or shed.get(item, 0) > 0:
                candidates.append(_candidate(farmer_op="PICKUP", item_index=V.entity_index(item)))

    return candidates


def is_legal_unit_action(
    farm: dict,
    shed: dict,
    seeds: dict,
    inventory: dict,
    pos: tuple[int, int],
    day: int,
    op_name: str,
    item_name: str | None,
    shed_capacity: int = 100,
) -> bool:
    """Return whether an intent is executable in the current shadow state.

    Args:
        farm: Current farm shadow state.
        shed: Current shed shadow state.
        seeds: Current seed shadow state.
        inventory: Inventory of the acting unit.
        pos: Current unit position as (x, y).
        day: Current day.
        op_name: Farmer operation name.
        item_name: Optional item selected by the operation.
        shed_capacity: Shed capacity.

    Returns:
        Whether the operation and item occur in the current legal candidates.
    """
    item_index = V.entity_index(item_name) if item_name is not None else None
    selected = tuple(_candidate(farmer_op=op_name, item_index=item_index).index)
    return any(
        tuple(candidate.index) == selected
        for candidate in legal_unit_actions(farm, shed, seeds, inventory, pos, day, shed_capacity)
    )


def legal_market_actions(
    farm: dict,
    shed: dict,
    market: dict,
    hire_mult: float = 1,
    shed_capacity: int = 100,
) -> list:
    """市場注文キューの1スロット分として、今合法な行動候補を列挙する。

    Args:
        farm: obs["farms"][player]。money, unlocked_quadrants, hires_today, handsを使う。
        shed: このプレイヤーの納屋(private["shed"])。SELL/BUY_ANIMAL等の容量判定に使う。
        market: obs["market"]。inventory/pricesを使う。
        hire_mult: farmHandCostMult。
        shed_capacity: 納屋の容量。

    Returns:
        list[SparseVector]: 合法な(op, item)候補のリスト。
    """
    money = farm["money"]
    candidates = []

    hire_cost = _fib(farm["hires_today"]) * hire_mult
    # market_orders.apply_hireは空きhandスロットが無ければ所持金不足と同様に無視する
    # (元のルールには無いMAX_HANDS上限)。
    if money >= hire_cost and len(farm["hands"]) < C.MAX_HANDS:
        candidates.append(_candidate(market_op="HIRE"))

    n_unlocked_extra = len(farm["unlocked_quadrants"]) - 1
    if n_unlocked_extra < len(P.LAND_PRICES) and money >= P.LAND_PRICES[n_unlocked_extra]:
        candidates.append(_candidate(market_op="BUY_LAND"))

    for i, crop in enumerate(C.CROPS):
        if has_executable_quantity("BUY_SEED", crop, farm, shed, market, shed_capacity):
            candidates.append(_candidate(market_op="BUY_SEED", item_index=V.ENTITY_ITEM[i]))

    for i, animal in enumerate(C.ANIMALS):
        if has_executable_quantity("BUY_ANIMAL", animal, farm, shed, market, shed_capacity):
            candidates.append(
                _candidate(market_op="BUY_ANIMAL", item_index=V.ENTITY_ITEM[C.N_PRODUCTS + i])
            )
    for item in ("WHEAT", "FERTILIZER"):
        # 表示価格(現在庫基準)ではなく、実際に1個買う時の価格(在庫-1基準、
        # market_lockstep._quote_and_commit参照)で判定する。表示価格だけで
        # 判定すると、実際は買えないのに候補に出て数量スロットが空になりうる。
        if has_executable_quantity("BUY_PRODUCT", item, farm, shed, market, shed_capacity):
            candidates.append(_candidate(market_op="BUY_PRODUCT", item_index=V.entity_index(item)))

    for item in C.PRODUCTS:
        if has_executable_quantity("SELL", item, farm, shed, market):
            candidates.append(_candidate(market_op="SELL", item_index=V.entity_index(item)))

    return candidates


def market_stop_candidate() -> V.SparseVector:
    """市場注文の決定を打ち切る(これ以上注文しない)候補。常に合法。

    legal_market_actionsが返す候補と違い、simulator側に対応する行動は無い
    (デコーダが可変長の市場注文リストの終わりを決めるための専用候補)。
    """
    sv = V.SparseVector()
    sv.add(V.ACTION_MARKET_STOP[0])
    return sv


def market_wait_candidate() -> V.SparseVector:
    """市場状態を変えず、次の市場注文スロットへ進む候補。"""
    sv = V.SparseVector()
    sv.add(V.ACTION_MARKET_WAIT[0])
    return sv


def quantity_candidates(max_quantity: int) -> list[V.SparseVector]:
    """(op, item)候補が確定した後、数量を選ぶためのスロットの候補を列挙する。

    量1個からmax_quantity個までを、それぞれ別々の添字(ACTION_QUANTITY_VALUE)を
    持つ候補として列挙する(vocab.ACTION_QUANTITY_VALUEのコメント参照)。これに
    加えて、正規化したlog(量)の連続値(ACTION_QUANTITY_CONTINUOUS)も同じ候補へ
    足す。量固有embeddingだけだと49と50が無関係な表現になり、大きい量ほど
    出現頻度が下がるぶん学習も遅くなる。ACTION_QUANTITY_VALUEという量ごとに
    一意な項(value=1.0)が同じ候補に既にあるため、この連続値を追加しても
    LayerNormでスケール情報が消える心配はない(PLAYER_MONEY_CONTINUOUSと同じ
    理由: 単独スケーリングではなく、固定項との比率として残る)。
    max_quantityはdecode_state.max_executable_quantityで求めた「今実行できる
    最大数量」を渡す想定(呼び出し側の責務。0以下なら空リストを返す=この
    (op, item)自体が候補にならないはず)。

    Returns:
        list[SparseVector]: 候補[i]が「量i+1」に対応する。長さmax_quantity
        (vocab.MAX_ACTION_QUANTITYで頭打ち)。
    """
    n = max(0, min(max_quantity, V.MAX_ACTION_QUANTITY))
    log_cap = math.log1p(V.MAX_ACTION_QUANTITY)
    candidates = []
    for k in range(1, n + 1):
        sv = V.SparseVector()
        sv.add(V.ACTION_QUANTITY_VALUE[k - 1])
        sv.add(V.ACTION_QUANTITY_CONTINUOUS[0], math.log1p(k) / log_cap)
        candidates.append(sv)
    return candidates


def requires_quantity(op_name: str, item_name: str | None, tile=None) -> bool:
    """このopを選んだ後、数量スロット(quantity_candidates)が必要かの判定基準を
    一箇所にまとめる(呼び出し側ごとに書くと対応がずれるため)。

    PLACEは動物配置(数量なし)と納屋落とし(数量あり)の2種があり、同じ
    item_nameでも今のタイル次第でどちらにもなりうる(is_animal_placement参照)
    ため、farmer/hand用の呼び出しはtileを渡すこと(marketのopにはタイルが
    無いため不要)。
    """
    if op_name == "PLACE":
        return item_name is not None and not is_animal_placement(item_name, tile)
    return op_name in ("PICKUP", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "SELL")


def _shape(func_code: int, x: float, t: float) -> np.float32:
    """market._shapeの純Python版(1つのfunc_code・スカラー入力用)。

    JAX側はjax_enable_x64を有効化していないためfloat32で計算している。ここを
    Pythonのfloat(64bit)で計算すると、丸め境界(round)付近でJAX版と1違う価格に
    なることがある(実測済み)。numpy.float32で全ての中間計算を揃えることでこれを防ぐ。
    """
    x = np.float32(max(0.0, x))
    t32 = np.float32(t)
    u = x / t32 if t > 0 else x
    if func_code == C.FUNC_LINEAR:
        return x
    if func_code == C.FUNC_SQ:
        return x * x
    if func_code == C.FUNC_SQRT:
        return np.sqrt(x)
    if func_code == C.FUNC_LOG:
        return np.log1p(x)
    if func_code == C.FUNC_LOG10:
        return np.log10(np.float32(1.0) + x)
    if func_code == C.FUNC_HINGE:
        if t > 0:
            return (
                u + np.float32(P.HINGE_GAIN) * np.maximum(np.float32(0.0), u - np.float32(1.0)) ** 2
            )
        return x
    return x


def _market_price_one(item_idx: int, inventory_value: float) -> int:
    """market.market_price_oneの純Python版。

    SELL/BUY_PRODUCTはロックステップで1単位ずつ価格が動く(market_lockstep.py参照)
    ため、n>1の注文を正しく評価するにはこの関数を1単位ごとに呼び直す必要がある。
    """
    inventory_value = np.float32(inventory_value)
    i0 = np.float32(P.MARKET_I0)
    below = inventory_value < i0
    base = np.float32(P.MARKET_BASE_PRICE[item_idx])
    t = np.float32(P.MARKET_T[item_idx])
    func = P.MARKET_BELOW_FUNC[item_idx] if below else P.MARKET_ABOVE_FUNC[item_idx]
    target = np.float32(
        P.MARKET_BELOW_TARGET[item_idx] if below else P.MARKET_ABOVE_TARGET[item_idx]
    )
    sign = np.float32(1.0) if below else np.float32(-1.0)
    x = (i0 - inventory_value) if below else (inventory_value - i0)

    amp = target * base / _shape(func, t, t)
    price = base + sign * amp * _shape(func, x, t)
    return max(P.PRICE_FLOOR, int(np.round(price)))


def _simulate_market_units(
    op_name: str,
    item_name: str,
    money: float,
    shed: dict,
    market_inventory: dict,
    shed_capacity: int,
    max_units: int,
) -> tuple[int, float, int, int]:
    """SELL/BUY_PRODUCTを1単位ずつmax_units回まで試行する
    (market_lockstep.pyの1プレイヤー分をPythonで再現)。

    「この(op, item)は最大何個までいけるか」(max_executable_quantity)と
    「実際に選ばれたn個を確定させる」(decode_state.commit_market_action)の両方で、
    価格が単位ごとに動くという同じ性質を扱うため、ロジックを1箇所にまとめている。

    Returns:
        (実際に成立した回数, 更新後のmoney, 更新後のshed在庫数, 更新後の市場在庫数)。
    """
    item_idx = C.PRODUCTS.index(item_name)
    shed_count = shed.get(item_name, 0)
    other_shed_total = sum(shed.values()) - shed_count
    inv = market_inventory[item_name]
    actual = 0

    for _ in range(max_units):
        if op_name == "SELL":
            if shed_count <= 0:
                break
            price = _market_price_one(item_idx, inv)
            money += price
            shed_count -= 1
            if price > 1:
                inv += 1
        else:  # BUY_PRODUCT
            price = _market_price_one(item_idx, inv - 1)
            if money < price or other_shed_total + shed_count >= shed_capacity:
                break
            money -= price
            shed_count += 1
            inv -= 1
        actual += 1

    return actual, money, shed_count, inv


def max_executable_quantity(
    op_name: str,
    item_name: str | None,
    farm: dict,
    shed: dict,
    market: dict,
    shed_capacity: int = 100,
    inventory: dict | None = None,
) -> int:
    """今の状態でこの(op, item)候補が実際に実行できる最大数量を計算する。

    legal_market_actionsの候補生成(数量0なら候補自体を出さない)、
    quantity_candidatesが列挙する候補の上限、およびdecode_state.commit_unit_action
    (PICKUP/PLACE納屋落とし)/commit_market_actionが実際に適用する数量の上限として、
    複数箇所から使う共通関数(呼び出し側ごとに再実装すると計算がずれるリスクがあるため)。

    Args:
        op_name: constants.FARMER_OP_NAMES(PICKUP/PLACE)またはconstants.MARKET_OP_NAMES。
        item_name: 対象品目名。HIRE/BUY_LANDには数量の概念が無いため使わない。
        farm: obs["farms"][player]相当。moneyを使う。
        shed: このプレイヤーの納屋。
        market: obs["market"]相当。BUY_PRODUCTでinventoryを使う。
        shed_capacity: 納屋の容量。
        inventory: このユニットの持ち物(private["inventories"][idx])。PLACE
            (納屋落とし)の時だけ必須。他のopでは不要。

    Returns:
        int。MAX_ACTION_QUANTITYで頭打ちにする。
    """
    if item_name is None:
        return 0

    if op_name in ("PICKUP", "SELL"):
        return min(shed.get(item_name, 0), V.MAX_ACTION_QUANTITY)

    if op_name == "PLACE":
        held = (inventory or {}).get(item_name, 0)
        room = max(shed_capacity - sum(shed.values()), 0)
        return min(held, room, V.MAX_ACTION_QUANTITY)

    if op_name == "BUY_SEED":
        cost = P.CROP_SEED_COST[C.CROPS.index(item_name)]
        return min(int(farm["money"] // cost), V.MAX_ACTION_QUANTITY)

    if op_name == "BUY_ANIMAL":
        cost = P.ANIMAL_COST[C.ANIMALS.index(item_name)]
        room = max(shed_capacity - sum(shed.values()), 0)
        return min(int(farm["money"] // cost), room, V.MAX_ACTION_QUANTITY)

    if op_name == "BUY_PRODUCT":
        actual, *_ = _simulate_market_units(
            op_name,
            item_name,
            farm["money"],
            shed,
            market["inventory"],
            shed_capacity,
            V.MAX_ACTION_QUANTITY,
        )
        return actual

    return 0


def has_executable_quantity(
    op_name: str,
    item_name: str | None,
    farm: dict,
    shed: dict,
    market: dict | None,
    shed_capacity: int = 100,
    inventory: dict | None = None,
) -> bool:
    """数量付き行動が1個以上成立するかを定数時間で判定する。

    合法な(op, item)候補の列挙では最大数量は不要である。特に市場価格を1個ずつ
    再計算する`max_executable_quantity`を候補ごとに呼ばないための軽量経路。
    """
    if item_name is None:
        return False

    if op_name in ("PICKUP", "SELL"):
        return shed.get(item_name, 0) > 0

    if op_name == "PLACE":
        return (inventory or {}).get(item_name, 0) > 0 and sum(shed.values()) < shed_capacity

    if op_name == "BUY_SEED":
        cost = P.CROP_SEED_COST[C.CROPS.index(item_name)]
        return farm["money"] >= cost

    if op_name == "BUY_ANIMAL":
        cost = P.ANIMAL_COST[C.ANIMALS.index(item_name)]
        return farm["money"] >= cost and sum(shed.values()) < shed_capacity

    if op_name == "BUY_PRODUCT":
        if market is None or sum(shed.values()) >= shed_capacity:
            return False
        item_idx = C.PRODUCTS.index(item_name)
        inventory_value = market["inventory"][item_name]
        price = _market_price_one(item_idx, inventory_value - 1)
        return farm["money"] >= price

    return False


def _fib(n: int) -> int:
    """fib(0)=1, fib(1)=1, fib(2)=2, ... (HIREのn人目のコスト倍率)。"""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a
