"""検証カバレッジを広げるための、全行動を意図的に踏みに行くエージェント。

random/starterエージェントは行動の一部(移動・WATER・HARVEST・BUY_SEED程度)しか
使わないため、動物系(BUILD_*・FEED・CARE・COLLECT_FERTILIZER・PLACE)・HIRE・
BUY_LAND・PICKUP/DROP・FERTILIZE・DIGがgolden trace検証で一度も踏まれていなかった。
このエージェントは各ユニットに役割を持たせ、必要な物を納屋から積んで目的地まで
運ぶ。

注意: FEED/FERTILIZE/PLACEはprivate["shed"]ではなく、ユニット自身の
private["inventories"][idx](farmerは0、handsは1始まり)を消費する
(元コードの_inv_take参照)。買った家畜や小屋の資材はまず納屋に入るので、
使うにはPICKUPで自分のinventoryへ積んでから運ぶ必要がある。
"""

import random

_RNG = random.Random(12345)

_ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
_BUILD_TOGGLE = False


def _is_shed_adjacent(pos, board_size):
    """posが納屋(盤面中央4マス)に隣接しているか。PICKUP/DROPが使える位置。"""
    half = board_size // 2
    return pos in ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def _step_toward(pos, target, board_size):
    """posからtargetへ1歩近づく移動opを返す(マンハッタン距離、障害物は考慮しない)。"""
    fx, fy = pos
    tx, ty = target
    dx, dy = tx - fx, ty - fy
    if dx != 0:
        return ["EAST"] if dx > 0 else ["WEST"]
    if dy != 0:
        return ["SOUTH"] if dy > 0 else ["NORTH"]
    return ["PASS"]


def _find_nearest(farm, pos, predicate):
    """predicateを満たす最も近いタイルの座標を盤面全体から探す(無ければNone)。"""
    board_size = len(farm["tiles"])
    best, best_dist = None, None
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if predicate(tile):
                dist = abs(x - pos[0]) + abs(y - pos[1])
                if best_dist is None or dist < best_dist:
                    best, best_dist = (x, y), dist
    return best


def _decide_unit_action(farm, private, idx, pos, board_size):
    """1ユニット(farmerまたは1体のhand)の行動を、現在のタイル/持ち物から決める。"""
    fx, fy = pos
    tile = farm["tiles"][fy][fx]
    shed = private["shed"]
    inv = private["inventories"][idx]

    # 動物のいる小屋: 給餌→世話→肥料回収→収穫の優先順で対応する
    if isinstance(tile, dict) and "animal" in tile:
        if not tile["fed_today"] and inv.get("WHEAT", 0) > 0:
            return ["FEED"]
        if not tile["cared_today"]:
            return ["CARE"]
        if tile["fertilizer_available"]:
            return ["COLLECT_FERTILIZER"]
        if tile["yield_units"] > 0:
            return ["HARVEST"]
    elif isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
        for animal, structure in _ANIMAL_STRUCTURE.items():
            if tile["kind"] == structure and inv.get(animal, 0) > 0:
                return ["PLACE", animal]
    # 植物: 施肥→水やり→収穫の優先順
    elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
        if inv.get("FERTILIZER", 0) > 0 and tile["fertilized_until_day"] < 0:
            return ["FERTILIZE"]
        if not tile["watered_today"]:
            return ["WATER"]
        if tile["yield_units"] > 0:
            return ["HARVEST"]
    elif isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]
    # 空きマス: 種があれば植える。無ければ交互にCOOP/PASTUREを建てる
    elif tile is None:
        for crop, count in private["seeds"].items():
            if count > 0:
                return ["PLANT", crop]
        if _RNG.random() < 0.08:
            global _BUILD_TOGGLE
            _BUILD_TOGGLE = not _BUILD_TOGGLE
            return ["BUILD_COOP"] if _BUILD_TOGGLE else ["BUILD_PASTURE"]

    # ここまでで何も出来なければ、積み荷を運ぶか納屋で積み下ろしする。
    # WHEAT/FERTILIZER/動物は「使うために運んでいる荷物」として扱い、
    # 目的地に着くまでDROPしない(そうしないとPICKUPした直後に同じマスで
    # またDROPしてしまい、いつまでも運べない)。
    for animal, structure in _ANIMAL_STRUCTURE.items():
        if inv.get(animal, 0) > 0:
            target = _find_nearest(
                farm,
                pos,
                lambda t, s=structure: (
                    isinstance(t, dict) and t.get("kind") == s and "animal" not in t
                ),
            )
            if target is not None:
                return _step_toward(pos, target, board_size)
    if inv.get("FERTILIZER", 0) > 0:
        target = _find_nearest(
            farm,
            pos,
            lambda t: (
                isinstance(t, dict) and t.get("kind") == "PLANT" and t["fertilized_until_day"] < 0
            ),
        )
        if target is not None:
            return _step_toward(pos, target, board_size)
    if inv.get("WHEAT", 0) > 0:
        target = _find_nearest(
            farm, pos, lambda t: isinstance(t, dict) and "animal" in t and not t["fed_today"]
        )
        if target is not None:
            return _step_toward(pos, target, board_size)

    sellable = {
        k: n for k, n in inv.items() if k not in _ANIMAL_STRUCTURE and k != "FERTILIZER" and n > 0
    }
    if sellable:
        if _is_shed_adjacent(pos, board_size):
            return ["DROP"]
        return _step_toward(pos, _nearest_shed_tile(pos, board_size), board_size)

    if _is_shed_adjacent(pos, board_size):
        for animal in _ANIMAL_STRUCTURE:
            if shed.get(animal, 0) > 0:
                return ["PICKUP", animal, 1]
        if shed.get("FERTILIZER", 0) > 0:
            return ["PICKUP", "FERTILIZER", 1]
        if shed.get("WHEAT", 0) > 1:  # 1枚は売却用に納屋へ残す
            return ["PICKUP", "WHEAT", 1]

    return [_RNG.choice(["NORTH", "SOUTH", "EAST", "WEST", "PASS"])]


def _nearest_shed_tile(pos, board_size):
    """納屋隣接4マスのうち、posから最も近い1マスを返す。"""
    half = board_size // 2
    candidates = ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))
    return min(candidates, key=lambda t: abs(t[0] - pos[0]) + abs(t[1] - pos[1]))


def coverage_agent(obs):
    """kaggle-environments用のエージェント本体。farmer/hands/marketの行動を返す。"""
    player = obs["player"]
    farm = obs["farms"][player]
    private = obs["private"]
    day = obs["day"]
    hour = obs["hour"]
    board_size = len(farm["tiles"])
    shed = private["shed"]

    market = []
    if day == 0 and hour == 0:
        market += [
            ["BUY_LAND"],
            ["BUY_SEED", "WHEAT", 3],
            ["BUY_SEED", "MELON", 1],
            ["BUY_ANIMAL", "GOOSE", 1],
            ["BUY_ANIMAL", "COW", 1],
            ["BUY_ANIMAL", "SHEEP", 1],
        ]
    if hour == 0:
        # hands は日をまたぐと解雇されるため、動物の世話役を毎日再雇用する。
        market.append(["HIRE"])
        market.append(["HIRE"])
        # FERTILIZEを確実に踏むため、COLLECT_FERTILIZER頼みにせず直接購入もする。
        market.append(["BUY_PRODUCT", "FERTILIZER", 1])
        if day in (3, 6):
            market.append(["BUY_LAND"])
        if day % 4 == 0 and shed.get("GOOSE", 0) == 0 and shed.get("COW", 0) == 0:
            market.append(["BUY_ANIMAL", "GOOSE", 1])
            market.append(["BUY_ANIMAL", "COW", 1])

    # 納屋の収穫物は随時売る(動物本体・飼料/施肥用に残したいWHEAT・FERTILIZERは除く)
    for item, n in shed.items():
        if (
            n > 0
            and item not in ("GOOSE", "COW", "SHEEP", "WHEAT", "FERTILIZER")
            and len(market) < 10
        ):
            market.append(["SELL", item, n])

    farmer_action = _decide_unit_action(farm, private, 0, tuple(farm["farmer"]), board_size)
    hands_actions = [
        _decide_unit_action(farm, private, i + 1, tuple(pos), board_size)
        for i, pos in enumerate(farm["hands"])
    ]

    return {"farmer": farmer_action, "hands": hands_actions, "market": market[:10]}
