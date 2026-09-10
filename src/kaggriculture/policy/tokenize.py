"""生のobservationを方策用SparseVector列へ変換する。

actorには公開情報と自分のprivateだけを渡す。非対称critic用の両者のprivateは
get_privileged_critic_inputで別系列にする。
"""

import math

from kaggriculture.policy import token_layout as L
from kaggriculture.policy import vocab as V
from kaggriculture.simulator import constants as C


def _norm(n: float, scale: float) -> float:
    """個数・日数等をだいたい[0, 1]付近に収める簡易正規化。"""
    return n / scale


def _norm_log(n: float, cap: float) -> float:
    """所持金など、桁が大きく変動する量をlog1pで[0, 1]付近に圧縮する。"""
    return math.log1p(max(n, 0)) / math.log1p(cap)


def _norm_clip(n: float, window: float) -> float:
    """「近いほど重要、遠ければ差を気にしなくてよい」量をwindowで頭打ちにして正規化する
    (例: 減衰までの残りターン数。0に近いほど緊急度が高いという意味を保つ)。"""
    return min(max(n, 0), window) / window


def _encode_tile(tile, day: int, step: int, is_farmer: bool, hand_count: int) -> V.SparseVector:
    """1マス分の生データをSparseVectorに変換する。"""
    sv = V.SparseVector()

    if tile is None:
        sv.add(V.TILE_KIND[C.TILE_EMPTY])
    elif tile == "LOCKED":
        sv.add(V.TILE_KIND[C.TILE_LOCKED])
    elif tile["kind"] == "WEED":
        sv.add(V.TILE_KIND[C.TILE_WEED])
    elif tile["kind"] == "PLANT":
        sv.add(V.TILE_KIND[C.TILE_PLANT])
        sv.add(V.entity_index(tile["crop"]))
        if tile["watered_today"]:
            sv.add(V.TILE_CARE_DONE_TODAY[0])
        if tile["fertilized_until_day"] >= day:
            sv.add(V.TILE_FERTILIZED_ACTIVE[0])
        sv.add(V.TILE_AGE[0], _norm(day - tile["planted_day"], 30))
        sv.add(V.TILE_YIELD_UNITS[0], _norm(tile["yield_units"], 6))
        sv.add(V.TILE_CONSECUTIVE_UNCARED[0], _norm(tile["consecutive_unwatered"], 2))
        if tile["max_lifespan_step"] >= 0:
            sv.add(
                V.TILE_LIFESPAN_REMAINING[0],
                _norm_clip(tile["max_lifespan_step"] - step, window=60),
            )
    else:  # COOP または PASTURE
        kind_idx = C.TILE_COOP if tile["kind"] == "COOP" else C.TILE_PASTURE
        sv.add(V.TILE_KIND[kind_idx])
        if tile.get("animal") is not None:
            sv.add(V.entity_index(tile["animal"]))
            if tile["fed_today"]:
                sv.add(V.TILE_CARE_DONE_TODAY[0])
            if tile["cared_today"]:
                sv.add(V.TILE_CARED_TODAY[0])
            if tile["fertilizer_available"]:
                sv.add(V.TILE_FERTILIZER_AVAILABLE[0])
            sv.add(V.TILE_AGE[0], _norm(day - tile["placed_day"], 30))
            sv.add(V.TILE_YIELD_UNITS[0], _norm(tile["yield_units"], 6))
            sv.add(V.TILE_CONSECUTIVE_UNCARED[0], _norm(tile["consecutive_unfed"], 2))
            sv.add(V.TILE_PENDING_CARE_BONUS[0], _norm_clip(tile["pending_care_bonus"], window=10))

    if is_farmer:
        sv.add(V.TILE_IS_FARMER[0])
    if hand_count:
        sv.add(V.TILE_HAND_COUNT[0], _norm_clip(hand_count, window=8))

    return sv


def _encode_board(farm, day: int, step: int) -> list[V.SparseVector]:
    """1プレイヤー分の盤面をSparseVectorのリストに変換する(y*board_size+x順)。"""
    board_size = len(farm["tiles"])
    hand_counts: dict[tuple[int, int], int] = {}
    for hx, hy in farm["hands"]:
        hand_counts[(hx, hy)] = hand_counts.get((hx, hy), 0) + 1
    fx, fy = farm["farmer"]

    tokens = []
    for y in range(board_size):
        for x in range(board_size):
            is_farmer = (x, y) == (fx, fy)
            hand_count = hand_counts.get((x, y), 0)
            tokens.append(_encode_tile(farm["tiles"][y][x], day, step, is_farmer, hand_count))
    return tokens


def _encode_player_info(farm) -> V.SparseVector:
    sv = V.SparseVector()
    sv.add(V.PLAYER_MONEY[0])
    money_norm = _norm_log(farm["money"], cap=200_000)
    # bucketの離散化誤差を連続値で補い、criticが終盤の金額差を読めるようにする。
    sv.add(V.PLAYER_MONEY_MAGNITUDE_BUCKET[V.bucket_index(money_norm)])
    sv.add(V.PLAYER_MONEY_CONTINUOUS[0], money_norm)
    for q in farm["unlocked_quadrants"]:
        sv.add(V.PLAYER_UNLOCKED_QUADRANT[C.QUADRANTS.index(q)])
    sv.add(V.PLAYER_HIRES_TODAY[0], _norm_clip(farm["hires_today"], window=16))
    return sv


def _add_counts(
    sv: V.SparseVector,
    items: dict,
    order: tuple,
    index_table,
    magnitude_table,
    normalize,
    continuous_table=None,
) -> None:
    """正の個数を存在マーカー、量bucket、任意の連続値として追加する。

    Args:
        sv: 追加先。
        items: 名前から個数への辞書。
        order: 名前の正規順序。
        index_table: 存在マーカーの語彙範囲。
        magnitude_table: 量bucketの語彙範囲。
        normalize: 数量の正規化関数。
        continuous_table: 品目別の連続値語彙範囲。
    """
    for name, n in items.items():
        if n > 0:
            idx = order.index(name)
            sv.add(index_table[idx])
            norm = normalize(n)
            bucket = V.bucket_index(norm)
            sv.add(magnitude_table[idx * V.N_MAGNITUDE_BUCKETS + bucket])
            if continuous_table is not None:
                sv.add(continuous_table[idx], norm)


def _add_shed_item_counts(sv: V.SparseVector, items: dict, cap: float = 100) -> None:
    """{品目名: 個数}をENTITY_ITEM語彙で追加する(shed/持ち物で共通)。"""
    _add_counts(
        sv,
        items,
        C.SHED_ITEMS,
        V.ENTITY_ITEM,
        V.ENTITY_MAGNITUDE_BUCKET,
        lambda n: _norm_log(n, cap=cap),
    )


def _encode_inventory_sum(inventories: list[dict]) -> V.SparseVector:
    """複数ユニット分の持ち物を合算して1トークンにする。"""
    totals: dict[str, int] = {}
    for inv in inventories:
        for item, n in inv.items():
            totals[item] = totals.get(item, 0) + n
    sv = V.SparseVector()
    _add_shed_item_counts(sv, totals)
    return sv


def _encode_market_inventory(market: dict) -> V.SparseVector:
    sv = V.SparseVector()
    _add_counts(
        sv,
        market["inventory"],
        C.PRODUCTS,
        V.ENTITY_ITEM,
        V.ENTITY_MAGNITUDE_BUCKET,
        lambda n: _norm_log(n, cap=12_000),
    )
    return sv


def _encode_market_price(market: dict) -> V.SparseVector:
    sv = V.SparseVector()
    # hinge型の価格曲線は品薄時に急騰しうる(線形正規化だと外れ値に弱い)ためlogで圧縮する。
    _add_counts(
        sv,
        market["prices"],
        C.PRODUCTS,
        V.ENTITY_ITEM,
        V.ENTITY_MAGNITUDE_BUCKET,
        lambda p: _norm_log(p, cap=2000),
    )
    return sv


def _encode_town(town: dict) -> V.SparseVector:
    counts: dict[str, int] = {}
    for shop in town["unlocked_shops"]:
        counts[shop] = counts.get(shop, 0) + 1
    sv = V.SparseVector()
    _add_counts(
        sv, counts, C.SHOPS, V.TOWN_SHOP, V.TOWN_SHOP_MAGNITUDE_BUCKET, lambda n: _norm(n, 8)
    )
    return sv


def _encode_turn(day: int, hour: int) -> V.SparseVector:
    sv = V.SparseVector()
    sv.add(V.TURN_DAY[0], _norm(day, 30))
    sv.add(V.TURN_HOUR[0], _norm(hour, 24))
    return sv


def encode_unit_context(inventory: dict) -> V.SparseVector:
    """現在のunitが持つ品目をDecoder入力へ加える。

    Args:
        inventory: unitの持ち物。

    Returns:
        unit固有のSparseVector。位置はmodel側で別途加える。
    """
    sv = V.SparseVector()
    sv.add(V.UNIT_CONTEXT_INVENTORY[0])  # ENTITY_ITEMが持ち物由来であることを示すマーカー
    _add_shed_item_counts(sv, inventory)
    return sv


def _encode_own_counters(counters: dict) -> V.SparseVector:
    """呼び出し側が保持する、自分のエピソード累積実績を符号化する。

    estimated項目の制約はepisode_history.pyを参照。
    """
    sv = V.SparseVector()
    _add_counts(
        sv,
        counters.get("produced", {}),
        C.PRODUCTS,
        V.OWN_PRODUCED,
        V.OWN_PRODUCED_MAGNITUDE_BUCKET,
        lambda n: _norm_log(n, cap=1000),
        continuous_table=V.OWN_PRODUCED_CONTINUOUS,
    )
    _add_counts(
        sv,
        counters.get("sold", {}),
        C.PRODUCTS,
        V.OWN_SOLD,
        V.OWN_SOLD_MAGNITUDE_BUCKET,
        lambda n: _norm_log(n, cap=1000),
        continuous_table=V.OWN_SOLD_CONTINUOUS,
    )
    _add_counts(
        sv,
        counters.get("estimated_bought_product", {}),
        C.PRODUCTS,
        V.OWN_ESTIMATED_BOUGHT_PRODUCT,
        V.OWN_ESTIMATED_BOUGHT_PRODUCT_MAGNITUDE_BUCKET,
        lambda n: _norm_log(n, cap=1000),
        continuous_table=V.OWN_ESTIMATED_BOUGHT_PRODUCT_CONTINUOUS,
    )
    _add_counts(
        sv,
        counters.get("estimated_revenue", {}),
        C.PRODUCTS,
        V.OWN_ESTIMATED_REVENUE,
        V.OWN_ESTIMATED_REVENUE_MAGNITUDE_BUCKET,
        lambda n: _norm_log(n, cap=500_000),
        continuous_table=V.OWN_ESTIMATED_REVENUE_CONTINUOUS,
    )
    for item, sold in counters.get("has_ever_sold", {}).items():
        if sold:
            sv.add(V.OWN_HAS_EVER_SOLD[C.PRODUCTS.index(item)])
    return sv


def _encode_seeds(seeds: dict) -> V.SparseVector:
    sv = V.SparseVector()
    _add_counts(
        sv,
        seeds,
        C.CROPS,
        V.ENTITY_ITEM,
        V.ENTITY_MAGNITUDE_BUCKET,
        lambda n: _norm_log(n, cap=100),
    )
    return sv


def get_encoder_input(
    obs: dict,
    turns_per_day: int = 24,
    counters: dict | None = None,
) -> list[V.SparseVector]:
    """player視点の観測を固定長token列へ符号化する。

    Args:
        obs: 生のobservation。
        turns_per_day: 1日当たりのターン数。
        counters: 自分のエピソード累積実績。

    Returns:
        公開情報、自分のprivate、履歴からなるtoken列。
    """
    player = obs["player"]
    opponent = 1 - player
    own_farm = obs["farms"][player]
    opp_farm = obs["farms"][opponent]
    day = obs["day"]
    step = day * turns_per_day + obs["hour"]

    tokens = _encode_board(own_farm, day, step) + _encode_board(opp_farm, day, step)
    tokens.append(_encode_player_info(own_farm))
    tokens.append(_encode_player_info(opp_farm))

    private = obs["private"]
    own_shed = V.SparseVector()
    _add_shed_item_counts(own_shed, private["shed"])
    tokens.append(own_shed)

    tokens.append(_encode_seeds(private["seeds"]))

    tokens.append(_encode_inventory_sum(private["inventories"]))

    tokens.append(_encode_market_inventory(obs["market"]))
    tokens.append(_encode_market_price(obs["market"]))
    tokens.append(_encode_town(obs["town"]))
    tokens.append(_encode_turn(day, obs["hour"]))
    tokens.append(_encode_own_counters(counters or {}))

    return tokens


def get_privileged_critic_input(
    obs: dict, opponent_private: dict
) -> tuple[list[V.SparseVector], list[int], list[bool]]:
    """両者のprivateを非対称critic用の固定長系列へ符号化する。

    Args:
        obs: player視点の観測。
        opponent_private: 相手の非公開状態。

    Returns:
        token列、unit位置ID、存在しないhandのpadding mask。

    Raises:
        ValueError: inventory数と公開unit数が一致しない場合。
    """
    player = obs["player"]
    farms = (obs["farms"][player], obs["farms"][1 - player])
    privates = (obs["private"], opponent_private)
    tokens: list[V.SparseVector] = []
    position_ids: list[int] = []
    padding_mask: list[bool] = []

    for farm, private in zip(farms, privates, strict=True):
        expected_inventories = 1 + len(farm["hands"])
        if len(private["inventories"]) != expected_inventories:
            raise ValueError(
                "private inventories must contain farmer followed by every active hand: "
                f"expected {expected_inventories}, got {len(private['inventories'])}"
            )
        if len(farm["hands"]) > C.MAX_HANDS:
            raise ValueError(f"farm has more than MAX_HANDS={C.MAX_HANDS}")

        shed = V.SparseVector()
        _add_shed_item_counts(shed, private["shed"])
        tokens.extend([shed, _encode_seeds(private["seeds"])])
        position_ids.extend([L.NO_POSITION, L.NO_POSITION])
        padding_mask.extend([False, False])

        unit_positions = [farm["farmer"], *farm["hands"]]
        for inventory, (x, y) in zip(private["inventories"], unit_positions, strict=True):
            unit_inventory = V.SparseVector()
            _add_shed_item_counts(unit_inventory, inventory)
            tokens.append(unit_inventory)
            position_ids.append(y * V.BOARD_SIZE + x)
            padding_mask.append(False)

        missing_hands = C.MAX_HANDS - len(farm["hands"])
        tokens.extend(V.SparseVector() for _ in range(missing_hands))
        position_ids.extend([L.NO_POSITION] * missing_hands)
        padding_mask.extend([True] * missing_hands)

    if len(tokens) != L.NUM_PRIVILEGED_TOKENS:
        raise AssertionError("privileged critic token layout mismatch")
    return tokens, position_ids, padding_mask
