"""自己回帰デコード中の仮状態をシミュレータと同じ順で更新する。

これにより、後続unitや市場注文の合法候補が先行決定を反映する。PLANTの全unit一括
ブロックは逐次再現できないため、種を予約して供給超過そのものを防ぐ。
"""

from kaggriculture.policy.torch.actions import (
    _fib,
    _market_price_one,
    _simulate_market_units,
    is_animal_placement,
    max_executable_quantity,
    requires_quantity,
)
from kaggriculture.rules import constants as C
from kaggriculture.rules import game_params as P

__all__ = [
    "commit_market_action",
    "commit_unit_action",
    "max_executable_quantity",
    "requires_quantity",
]


def _harvest_product_name(tile: dict) -> str | None:
    """HARVESTで実際に増える品目名。植物ならcrop自身、動物ならその生産物
    (COW→MILK等。game_params.ANIMAL_PRODUCT_IDX参照)。どちらでもなければNone。
    """
    if tile.get("kind") == "PLANT":
        return tile["crop"]
    animal = tile.get("animal")
    if animal is not None:
        return C.PRODUCTS[P.ANIMAL_PRODUCT_IDX[C.ANIMALS.index(animal)]]
    return None


def commit_unit_action(
    farm: dict,
    shed: dict,
    seeds: dict,
    op_name: str,
    item_name: str | None,
    pos: tuple[int, int],
    day: int,
    n: int = 1,
    shed_capacity: int = 100,
    inventory: dict | None = None,
    turns_per_day: int = 24,
) -> dict[str, dict[str, int]]:
    """1ユニット分の確定した行動を、次スロットの合法候補計算用の状態に反映する。

    farm["tiles"]・shed・seedsを直接書き換える(呼び出し側がターン開始時点の
    コピーを渡す前提。元のobservationを直接渡すと壊れるので注意)。

    Returns:
        dict[str, dict[str, int]]: HARVEST/COLLECT_FERTILIZERの時だけ
        {"produced": {品目名: 実際に生産された量}}を返す(他のopでは空dict。
        commit_market_actionと同じ「カテゴリ→{品目: 量}」の形。episode_history.
        compute_turn_deltasが積み上げる)。「生産」はcrop_actions.apply_harvest/
        animal_actions.apply_collect_fertilizerがinventoryへ加える量で定義する
        (shedへの搬入や販売とは別の、最も早い時点の量)。

    Args:
        farm: obs["farms"][player]相当(tilesを書き換える)。
        shed: このプレイヤーの納屋(PICKUP/DROP/PLACE納屋落としで書き換える)。
        seeds: このプレイヤーの残り種(PLANTで書き換える)。
        op_name: 確定したconstants.FARMER_OP_NAMES。
        item_name: 対象品目名(該当opのみ)。PLANTなら作物名、PLACEなら動物名
            または品目名。
        pos: (x, y) このユニットの現在位置。
        day: 現在の日。PLANT/PLACE(動物配置)がplanted_day/placed_dayに使う
            (actions.legal_unit_actionsのHARVEST成熟判定に必要)。
        n: PICKUP/PLACE(納屋落とし)の要求数量(inventory_actions.apply_pickup/
            animal_actions.apply_placeの`n`引数と同じ)。PLACEでは、呼び出し側で
            既にこのユニットの持ち物数までクランプした値を渡すこと(このユニットの
            持ち物は他ユニットの判定に使われないためここでは保持しておらず、
            クランプできない)。
        shed_capacity: 納屋の容量(PLACE納屋落とし/DROPの空き容量クランプに使う)。
        inventory: このユニットの持ち物(private["inventories"][idx])。DROPの
            時だけ必須(全品目をshedへ移すため)。他のopでは不要。
        turns_per_day: 1日のターン数。PLANT(一発収穫型)のmax_lifespan_step計算に使う。
    """
    fx, fy = pos
    tile = farm["tiles"][fy][fx]
    produced: dict[str, int] = {}

    if op_name == "PICKUP" and item_name is not None:
        taken = min(max(n, 0), shed.get(item_name, 0))
        shed[item_name] = shed.get(item_name, 0) - taken

    elif op_name == "DROP":
        if inventory is None:
            raise ValueError("commit_unit_action: DROP requires inventory")
        # inventory_actions.dump_inventoryと同じ順序・クランプで、持ち物を
        # 全品目分shedへ移す(空き容量を使い切ったら以降の品目は入らない)。
        shed_total = sum(shed.values())
        for item in C.SHED_ITEMS:
            held = inventory.get(item, 0)
            if held <= 0:
                continue
            added = min(held, max(shed_capacity - shed_total, 0))
            shed[item] = shed.get(item, 0) + added
            shed_total += added

    elif op_name == "PLACE" and item_name is not None:
        # PLACEの意味は品目名だけでなく、現在のタイルにも依存する。
        if is_animal_placement(item_name, tile):
            farm["tiles"][fy][fx] = {
                "kind": tile["kind"],
                "animal": item_name,
                "fed_today": False,
                "cared_today": False,
                "fertilizer_available": False,
                "consecutive_unfed": 0,
                "pending_care_bonus": 0,
                "yield_units": 0,
                "placed_day": day,
            }
        else:
            if inventory is None:
                raise ValueError("commit_unit_action: PLACE(納屋落とし) requires inventory")
            shed_room = max(shed_capacity - sum(shed.values()), 0)
            taken = min(max(n, 0), inventory.get(item_name, 0), shed_room)
            shed[item_name] = shed.get(item_name, 0) + taken

    elif op_name == "PLANT" and item_name is not None:
        seeds[item_name] = seeds.get(item_name, 0) - 1
        # crop_actions.apply_plant参照: 一発収穫型は植付直後にyield_units=1・
        # max_lifespan_stepが有限値になる(=枯れるまでの上限を決める)。継続収穫型は
        # どちらも0/-1のまま(日次更新側で管理する)。
        crop_idx = C.CROPS.index(item_name)
        is_ongoing = P.CROP_IS_ONGOING[crop_idx]
        max_yield_day = P.CROP_MAX_YIELD_DAY[crop_idx]
        farm["tiles"][fy][fx] = {
            "kind": "PLANT",
            "crop": item_name,
            "watered_today": False,
            "consecutive_unwatered": 1,
            "fertilized_until_day": -1,
            "yield_units": 0 if is_ongoing else 1,
            "planted_day": day,
            "max_lifespan_step": -1 if is_ongoing else (day + max_yield_day + 1) * turns_per_day,
        }

    elif op_name == "BUILD_COOP":
        farm["tiles"][fy][fx] = {"kind": "COOP"}

    elif op_name == "BUILD_PASTURE":
        farm["tiles"][fy][fx] = {"kind": "PASTURE"}

    elif op_name == "DIG":
        farm["tiles"][fy][fx] = None

    elif op_name == "WATER":
        tile["watered_today"] = True

    elif op_name == "HARVEST":
        # 一発収穫型の作物(crop_actions.apply_harvest参照)は収穫後にタイルが
        # 更地に戻る。継続収穫型・動物は(yield_unitsをゼロに戻すだけで)残る。
        harvested_name = _harvest_product_name(tile)
        harvested_amount = tile.get("yield_units", 0)
        if tile.get("kind") == "PLANT" and not P.CROP_IS_ONGOING[C.CROPS.index(tile["crop"])]:
            farm["tiles"][fy][fx] = None
        else:
            tile["yield_units"] = 0
        if harvested_name is not None and harvested_amount > 0:
            produced[harvested_name] = produced.get(harvested_name, 0) + harvested_amount

    elif op_name == "FEED":
        tile["fed_today"] = True

    elif op_name == "CARE":
        tile["cared_today"] = True

    elif op_name == "COLLECT_FERTILIZER":
        tile["fertilizer_available"] = False
        produced["FERTILIZER"] = produced.get("FERTILIZER", 0) + 1

    elif op_name == "FERTILIZE":
        tile["fertilized_until_day"] = 10**9  # 他ユニットからは「有効中」に見えれば十分

    return {"produced": produced} if produced else {}


def commit_market_action(
    farm: dict,
    shed: dict,
    market: dict,
    op_name: str,
    item_name: str | None,
    n: int,
    shed_capacity: int = 100,
    hire_mult: float = 1,
) -> dict[str, dict[str, int | float]]:
    """1件の確定した市場注文を、次注文の合法候補計算用の状態に反映する。

    farm(money, hires_today, unlocked_quadrants)・shed・market(inventory/prices)を
    直接書き換える。nはmax_executable_quantityで求めた上限以下であることを
    呼び出し側が保証する前提(候補として存在しない数量を渡さない)。

    SELL/BUY_PRODUCTの相手プレイヤーの同ターンの市場注文は、行動を決める時点
    では分からない情報として扱う(推論時には得られない情報を学習時にだけ使うと、
    教師強制と自走時の挙動がずれる原因になる)ため、相手が同じ品目を同ターンに
    売買している場合、ここで計算した価格は実際の結果と一致しないことがある
    (既知の限界。market_lockstep.pyが両プレイヤーのSELL/BUY_PRODUCT(WHEAT・
    FERTILIZERのみ)を1単位ずつ同時に処理し、価格が互いの注文に応じて動くため、
    自分の注文だけを単独再生しても値が正確に一致するとは限らない)。

    Returns:
        dict: BUY_PRODUCT/SELLの時だけ{"estimated_bought_product": {item: n}}
        または{"sold": {item: n}, "estimated_revenue": {item: money}}を返す
        (他のopは空dict)。要求量(n)ではなく、shed容量・所持金でクランプされた
        決済量(shed/moneyの前後差分)を返すが、上記の理由によりBUY_PRODUCTの
        数量とSELLの収益は相手の同時注文次第で実際の値と異なりうる
        (estimated_接頭辞の由来)。SELLの個数(sold)自体はshed在庫のみで成立可否
        が決まり価格に依存しないため、こちらは正確な値になる。

    Args:
        farm: obs["farms"][player]相当(money, hires_today, unlocked_quadrantsを書き換える)。
        shed: このプレイヤーの納屋(SELL/BUY_PRODUCT/BUY_ANIMALで書き換える)。
        market: obs["market"]相当(inventory/pricesを書き換える)。
        op_name: 確定したconstants.MARKET_OP_NAMES。
        item_name: 対象品目名(HIRE/BUY_LANDはNone)。
        n: 確定した数量(HIRE/BUY_LANDは無視)。max_executable_quantity以下であること。
        shed_capacity: 納屋の容量。
        hire_mult: farmHandCostMult(HIREのコスト倍率)。
    """
    if op_name == "HIRE":
        cost = _fib(farm["hires_today"]) * hire_mult
        farm["money"] -= cost
        farm["hires_today"] += 1
        # unit行動は市場注文より前に終わるため、次注文の人数判定にだけ使う。
        farm["hands"] = [*farm["hands"], (0, 0)]

    elif op_name == "BUY_LAND":
        n_unlocked_extra = len(farm["unlocked_quadrants"]) - 1
        farm["money"] -= P.LAND_PRICES[n_unlocked_extra]
        # 後続注文は解放区画の数だけを参照する。
        farm["unlocked_quadrants"] = [*farm["unlocked_quadrants"], "_"]

    elif op_name == "BUY_SEED" and item_name is not None:
        cost = P.CROP_SEED_COST[C.CROPS.index(item_name)]
        actual = min(max(n, 0), int(farm["money"] // cost))
        farm["money"] -= cost * actual

    elif op_name == "BUY_ANIMAL" and item_name is not None:
        cost = P.ANIMAL_COST[C.ANIMALS.index(item_name)]
        room = max(shed_capacity - sum(shed.values()), 0)
        actual = min(max(n, 0), int(farm["money"] // cost), room)
        farm["money"] -= cost * actual
        shed[item_name] = shed.get(item_name, 0) + actual

    elif op_name in ("BUY_PRODUCT", "SELL") and item_name is not None:
        item_idx = C.PRODUCTS.index(item_name)
        prev_money = farm["money"]
        prev_shed_count = shed.get(item_name, 0)
        _, new_money, new_shed_count, new_inv = _simulate_market_units(
            op_name, item_name, farm["money"], shed, market["inventory"], shed_capacity, max(n, 0)
        )
        farm["money"] = new_money
        shed[item_name] = new_shed_count
        market["inventory"][item_name] = new_inv
        market["prices"][item_name] = _market_price_one(item_idx, new_inv)

        if op_name == "BUY_PRODUCT":
            bought = new_shed_count - prev_shed_count
            if bought > 0:
                return {"estimated_bought_product": {item_name: bought}}
        else:  # SELL
            sold = prev_shed_count - new_shed_count
            if sold > 0:
                return {
                    "sold": {item_name: sold},
                    "estimated_revenue": {item_name: new_money - prev_money},
                }

    return {}
