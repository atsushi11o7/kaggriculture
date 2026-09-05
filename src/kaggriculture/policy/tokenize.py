"""kaggle-environmentsの生observation(1ターン・1プレイヤー視点)を、
policy.vocabの語彙(トークン構成はvocab.py参照)に沿ったSparseVectorの列に変換する。

相手の非公開情報(shed/seeds/inventories)は元々observationに含まれない
(kaggle-environments側で既にマスクされている)ため、相手のshedトークンは
常に空になる。
"""

import math

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
        sv.add(V.TILE_CROP[C.CROPS.index(tile["crop"])])
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
            sv.add(V.TILE_ANIMAL[C.ANIMALS.index(tile["animal"])])
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
    sv.add(V.PLAYER_MONEY[0], _norm_log(farm["money"], cap=200_000))
    for q in farm["unlocked_quadrants"]:
        sv.add(V.PLAYER_UNLOCKED_QUADRANT[C.QUADRANTS.index(q)])
    sv.add(V.PLAYER_HIRES_TODAY[0], _norm_clip(farm["hires_today"], window=16))
    return sv


def _add_counts(sv: V.SparseVector, items: dict, order: tuple, index_table, normalize) -> None:
    """{名前: 個数} 形式の辞書のうち、値が正のものだけをsvに追加する共通処理。

    Args:
        sv: 追加先のSparseVector。
        items: {名前: 個数}。
        order: indexへの変換に使う並び順(constants.CROPS等)。
        index_table: orderのインデックスに対応する語彙添字の範囲。
        normalize: 個数を[0, 1]付近に正規化する関数。
    """
    for name, n in items.items():
        if n > 0:
            sv.add(index_table[order.index(name)], normalize(n))


def _encode_inventory_sum(inventories: list[dict]) -> V.SparseVector:
    """複数ユニット分の持ち物を合算して1トークンにする。"""
    totals: dict[str, int] = {}
    for inv in inventories:
        for item, n in inv.items():
            totals[item] = totals.get(item, 0) + n
    sv = V.SparseVector()
    _add_counts(sv, totals, C.SHED_ITEMS, V.INVENTORY_ITEM, lambda n: _norm(n, 100))
    return sv


def _encode_market(market: dict) -> V.SparseVector:
    sv = V.SparseVector()
    _add_counts(sv, market["inventory"], C.PRODUCTS, V.MARKET_PRODUCT, lambda n: _norm(n, 12_000))
    # hinge型の価格曲線は品薄時に急騰しうる(線形正規化だと外れ値に弱い)ためlogで圧縮する。
    _add_counts(sv, market["prices"], C.PRODUCTS, V.MARKET_PRICE, lambda p: _norm_log(p, cap=2000))
    return sv


def _encode_town(town: dict) -> V.SparseVector:
    counts: dict[str, int] = {}
    for shop in town["unlocked_shops"]:
        counts[shop] = counts.get(shop, 0) + 1
    sv = V.SparseVector()
    _add_counts(sv, counts, C.SHOPS, V.TOWN_SHOP, lambda n: _norm(n, 8))
    return sv


def _encode_turn(day: int, hour: int) -> V.SparseVector:
    sv = V.SparseVector()
    sv.add(V.TURN_DAY[0], _norm(day, 30))
    sv.add(V.TURN_HOUR[0], _norm(hour, 24))
    return sv


def get_encoder_input(obs: dict, turns_per_day: int = 24) -> list[V.SparseVector]:
    """1ターン分の観測(obs["player"]視点)から、盤面トークン列全体を作る。

    Args:
        obs: kaggle-environmentsの生observation(このプレイヤー視点。privateは
            自分の分だけが実数値、相手の分はそもそも含まれない)。
        turns_per_day: 通しステップ数の計算に使う(configurationのturnsPerDay)。

    Returns:
        トークン列。自分の盤面100+相手の盤面100+player_info×2+shed×2
        (相手は常に空)+seeds+inventory+market+town+turnの順。
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
    _add_counts(own_shed, private["shed"], C.SHED_ITEMS, V.SHED_ITEM, lambda n: _norm(n, 100))
    tokens.append(own_shed)
    tokens.append(V.SparseVector())  # 相手のshedは常に不明

    own_seeds = V.SparseVector()
    _add_counts(own_seeds, private["seeds"], C.CROPS, V.SEED_CROP, lambda n: _norm_log(n, cap=100))
    tokens.append(own_seeds)

    tokens.append(_encode_inventory_sum(private["inventories"]))
    tokens.append(_encode_market(obs["market"]))
    tokens.append(_encode_town(obs["town"]))
    tokens.append(_encode_turn(day, obs["hour"]))

    return tokens
