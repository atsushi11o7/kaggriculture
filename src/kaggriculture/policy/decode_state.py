"""デコード中(farmer→hands→市場注文の自己回帰生成で各スロットを確定するたび)に
farm/shed/market状態を仮想的に更新し、次スロットの合法候補が「直前までに確定した
決定」を正しく反映するようにする。

actions.legal_unit_actions/legal_market_actionsは、1スロット単体の合法手を
「渡された時点の状態」から計算するだけなので、呼び出し側が状態を更新せずに
使い回すと、shedの在庫を複数ユニットが二重に取り合う等の不整合が起こる。
シミュレータ自身もfarmer→hand0→hand1…(player_turn.py、shedとtileを共有しながら
jax.lax.while_loopで逐次処理)、市場注文も提出順に1件ずつ(market_queue.py、
moneyとshedを引き継ぎながらjax.lax.scanで逐次処理)という順序で処理しており、
このモジュールはその逐次処理を生のkaggle-environments observation(dict)の上で
Pythonのまま再現する。1ユニット/1市場注文の行動が確定するたびにcommit_unit_action/
commit_market_actionを呼び、その効果を反映した状態から次スロットの
actions.legal_*_actionsを呼び直す、という使い方を想定する。

PLANTの種切れは元シミュレータでは「そのターンの全ユニットの要求を集計し、
在庫を超えた作物は全要求を一括ブロックする」というターン全体のグローバルな
制約であり(crop_actions.compute_plant_block参照)、逐次処理では再現できない。
ここでは「そのターンで既に確定した同じ作物への要求数」を仮想的に種から
差し引いていくことで、供給超過そのものを起こさせない(=ブロックが発動する
状況を作らない)方針を取る。
"""

import math

from kaggriculture.policy.actions import _fib
from kaggriculture.simulator import constants as C
from kaggriculture.simulator import game_params as P

# このopが確定したとき、shedとtileのどちらを更新する必要があるか(actions.pyの
# legal_unit_actionsが読む条件のうち、他ユニット/後続スロットの判定に影響し
# うるものだけを反映する。個々のユニットのinventoryは、同一ターン内で他ユニットの
# 判定には使われない[player_turn.pyが各ユニットの元のinventoryをそのまま読むため]
# ので、ここでは更新しない)。
_TILE_TO_EMPTY_OPS = frozenset({"DIG"})


def commit_unit_action(
    farm: dict,
    shed: dict,
    seeds: dict,
    op_name: str,
    item_name: str | None,
    pos: tuple[int, int],
    n: int = 1,
    shed_capacity: int = 100,
) -> None:
    """1ユニット分の確定した行動を、次スロットの合法候補計算用の状態に反映する。

    farm["tiles"]・shed・seedsを直接書き換える(呼び出し側がターン開始時点の
    コピーを渡す前提。元のobservationを直接渡すと壊れるので注意)。

    Args:
        farm: obs["farms"][player]相当(tilesを書き換える)。
        shed: このプレイヤーの納屋(PICKUP/DROP/PLACE納屋落としで書き換える)。
        seeds: このプレイヤーの残り種(PLANTで書き換える)。
        op_name: 確定したconstants.FARMER_OP_NAMES。
        item_name: 対象品目名(該当opのみ)。PLANTなら作物名、PLACEなら動物名
            または品目名。
        pos: (x, y) このユニットの現在位置。
        n: PICKUP/PLACE(納屋落とし)の要求数量(inventory_actions.apply_pickup/
            animal_actions.apply_placeの`n`引数と同じ)。PLACEでは、呼び出し側で
            既にこのユニットの持ち物数までクランプした値を渡すこと(このユニットの
            持ち物は他ユニットの判定に使われないためここでは保持しておらず、
            クランプできない)。DROPは全品目分をまとめて移すため、呼び出し側が
            このユニットの持ち物をshedへ直接加算すること(この関数では扱わない)。
        shed_capacity: 納屋の容量(PLACE納屋落としの空き容量クランプに使う)。
    """
    fx, fy = pos
    tile = farm["tiles"][fy][fx]

    if op_name == "PICKUP" and item_name is not None:
        taken = min(max(n, 0), shed.get(item_name, 0))
        shed[item_name] = shed.get(item_name, 0) - taken

    elif op_name == "DROP":
        # DROPは持ち物を全品目分まとめて納屋へ移すが、ここでは合法候補の再計算に
        # 必要な「shedの在庫が増える」効果だけを、呼び出し側から渡された
        # inventoryを使わずには反映できない。呼び出し側でinventoryの内容を
        # 直接shedへ加算してから、この関数を呼ぶ前後で処理すること
        # (legal_unit_actions自体はshedの正確な量ではなくn>0しか見ないため、
        # ここでは「何か増えた」ことまでは要求しない)。
        pass

    elif op_name == "PLACE" and item_name is not None:
        if item_name in C.ANIMALS:
            # 動物配置側。空の対応する小屋であることはactions.py側で保証済み。
            farm["tiles"][fy][fx] = {
                "kind": tile["kind"],
                "animal": item_name,
                "fed_today": False,
                "cared_today": False,
                "fertilizer_available": False,
                "consecutive_unfed": 0,
                "pending_care_bonus": 0,
                "yield_units": 0,
                "placed_day": None,
            }
        else:
            # 納屋落とし側。呼び出し側で持ち物数までクランプ済みのnを、さらに
            # 納屋の空き容量でクランプする(animal_actions.apply_placeのn_take参照)。
            shed_room = max(shed_capacity - sum(shed.values()), 0)
            taken = min(max(n, 0), shed_room)
            shed[item_name] = shed.get(item_name, 0) + taken

    elif op_name == "PLANT" and item_name is not None:
        seeds[item_name] = seeds.get(item_name, 0) - 1
        farm["tiles"][fy][fx] = {
            "kind": "PLANT",
            "crop": item_name,
            "watered_today": False,
            "consecutive_unwatered": 0,
            "fertilized_until_day": -1,
            "yield_units": 0,
            "planted_day": None,
            "max_lifespan_step": -1,
        }

    elif op_name == "BUILD_COOP":
        farm["tiles"][fy][fx] = {"kind": "COOP"}

    elif op_name == "BUILD_PASTURE":
        farm["tiles"][fy][fx] = {"kind": "PASTURE"}

    elif op_name in _TILE_TO_EMPTY_OPS:
        farm["tiles"][fy][fx] = None

    elif op_name == "WATER":
        tile["watered_today"] = True

    elif op_name == "HARVEST":
        tile["yield_units"] = 0

    elif op_name == "FEED":
        tile["fed_today"] = True

    elif op_name == "CARE":
        tile["cared_today"] = True

    elif op_name == "COLLECT_FERTILIZER":
        tile["fertilizer_available"] = False

    elif op_name == "FERTILIZE":
        tile["fertilized_until_day"] = 10**9  # 他ユニットからは「有効中」に見えれば十分


def _shape(func_code: int, x: float, t: float) -> float:
    """market._shapeの純Python版(1つのfunc_code・スカラー入力用)。"""
    x = max(0.0, x)
    u = x / t if t > 0 else x
    if func_code == C.FUNC_LINEAR:
        return x
    if func_code == C.FUNC_SQ:
        return x * x
    if func_code == C.FUNC_SQRT:
        return math.sqrt(x)
    if func_code == C.FUNC_LOG:
        return math.log1p(x)
    if func_code == C.FUNC_LOG10:
        return math.log10(1.0 + x)
    if func_code == C.FUNC_HINGE:
        return u + P.HINGE_GAIN * max(0.0, u - 1.0) ** 2 if t > 0 else x
    return x


def _market_price_one(item_idx: int, inventory_value: float) -> int:
    """market.market_price_oneの純Python版。

    SELL/BUY_PRODUCTはロックステップで1単位ずつ価格が動く(market_lockstep.py参照)
    ため、n>1の注文を正しく評価するにはこの関数を1単位ごとに呼び直す必要がある。
    """
    below = inventory_value < P.MARKET_I0
    base = P.MARKET_BASE_PRICE[item_idx]
    t = P.MARKET_T[item_idx]
    func = P.MARKET_BELOW_FUNC[item_idx] if below else P.MARKET_ABOVE_FUNC[item_idx]
    target = P.MARKET_BELOW_TARGET[item_idx] if below else P.MARKET_ABOVE_TARGET[item_idx]
    sign = 1.0 if below else -1.0
    x = (P.MARKET_I0 - inventory_value) if below else (inventory_value - P.MARKET_I0)

    amp = target * base / _shape(func, t, t)
    price = base + sign * amp * _shape(func, x, t)
    return max(P.PRICE_FLOOR, round(price))


def commit_market_action(
    farm: dict,
    shed: dict,
    market: dict,
    op_name: str,
    item_name: str | None,
    n: int,
    shed_capacity: int = 100,
) -> None:
    """1件の確定した市場注文を、次注文の合法候補計算用の状態に反映する。

    farm(money, hires_today, unlocked_quadrants)・shed・market["inventory"]を
    直接書き換える。

    SELL/BUY_PRODUCTはmarket_lockstep.pyと同じ「1単位ずつ価格を計算し直す」
    処理を自分のnユニット分だけ再現する。ただし実際のロックステップは両
    プレイヤーの注文が同じmarket在庫を同時に奪い合う処理であり、この関数は
    自分の注文だけを反映する。相手の同ターンの市場注文はこの時点では分からない
    情報として扱う(たとえリプレイから分かっても、推論時には得られない情報を
    学習時にだけ使うと、教師強制と自走時の挙動がずれる原因になる)ため、相手が
    同じ品目を同ターンに売買している場合、ここで計算した価格は実際の結果と
    一致しないことがある(既知の限界)。

    Args:
        farm: obs["farms"][player]相当(money, hires_today, unlocked_quadrantsを書き換える)。
        shed: このプレイヤーの納屋(SELL/BUY_PRODUCT/BUY_ANIMALで書き換える)。
        market: obs["market"]相当(inventory/pricesを書き換える)。
        op_name: 確定したconstants.MARKET_OP_NAMES。
        item_name: 対象品目名(HIRE/BUY_LANDはNone)。
        n: 要求数量(HIRE/BUY_LANDは無視)。実際に確定した数はmoney/shed容量/
            市場在庫が尽きた時点で打ち切る(market_lockstepと同じ)。
        shed_capacity: 納屋の容量。
    """
    if op_name == "HIRE":
        cost = _fib(farm["hires_today"])
        farm["money"] -= cost
        farm["hires_today"] += 1

    elif op_name == "BUY_LAND":
        n_unlocked_extra = len(farm["unlocked_quadrants"]) - 1
        farm["money"] -= P.LAND_PRICES[n_unlocked_extra]
        # unlocked_quadrantsの実際の値(どの区画か)は合法候補の判定に使われない
        # (len()しか見ない)ため、ダミー値を足すだけでよい。
        farm["unlocked_quadrants"] = [*farm["unlocked_quadrants"], "_"]

    elif op_name == "BUY_SEED" and item_name is not None:
        cost = P.CROP_SEED_COST[C.CROPS.index(item_name)]
        farm["money"] -= cost * n

    elif op_name == "BUY_ANIMAL" and item_name is not None:
        cost = P.ANIMAL_COST[C.ANIMALS.index(item_name)]
        farm["money"] -= cost * n
        shed[item_name] = shed.get(item_name, 0) + n

    elif op_name == "BUY_PRODUCT" and item_name is not None:
        item_idx = C.PRODUCTS.index(item_name)
        for _ in range(n):
            price = _market_price_one(item_idx, market["inventory"][item_name] - 1)
            if farm["money"] < price or sum(shed.values()) >= shed_capacity:
                break
            farm["money"] -= price
            shed[item_name] = shed.get(item_name, 0) + 1
            market["inventory"][item_name] -= 1

    elif op_name == "SELL" and item_name is not None:
        item_idx = C.PRODUCTS.index(item_name)
        for _ in range(n):
            if shed.get(item_name, 0) <= 0:
                break
            price = _market_price_one(item_idx, market["inventory"][item_name])
            farm["money"] += price
            shed[item_name] -= 1
            if price > 1:
                market["inventory"][item_name] += 1
