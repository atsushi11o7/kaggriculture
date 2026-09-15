"""Model-independent sequential candidate generation for replay and GBDT data."""

from __future__ import annotations

from dataclasses import dataclass

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.torch import actions as A
from kaggriculture.policy.torch import decode as DS
from kaggriculture.policy.torch import tokenize
from kaggriculture.rules import constants as C


def _entry_from(op_name, item_name, n, force_quantity=False):
    """行動エントリを作る。市場注文では数量1も省略できない。"""
    entry = [op_name]
    if item_name is not None:
        entry.append(item_name)
        if n != 1 or force_quantity:
            entry.append(n)
    return entry


def _item_name(sv: V.SparseVector) -> str | None:
    """候補SparseVectorからENTITY_ITEMの品目名を取り出す(無ければNone)。"""
    for idx in sv.index:
        if V.ENTITY_ITEM.start <= idx < V.ENTITY_ITEM.stop:
            return C.SHED_ITEMS[idx - V.ENTITY_ITEM.start]
    return None


def _op_name(sv: V.SparseVector, op_table: range, op_names: tuple) -> str | None:
    """候補SparseVectorからop名を取り出す(市場のWAIT/STOP候補ならNone)。"""
    for idx in sv.index:
        if op_table.start <= idx < op_table.stop:
            return op_names[idx - op_table.start]
    return None


class _TeacherForceChooser:
    """保存済み行動に対応する候補indexを順に返す。"""

    def __init__(self, action: dict):
        self._farmer = list(action["farmer"])
        self._hands = [list(h) for h in action["hands"]]
        self._market = [list(m) for m in action["market"]]
        self._hand_cursor = 0
        self._market_cursor = 0
        self._pending_quantity: int | None = None

    def choose(self, cands: list, slot_kind: str) -> int:
        if slot_kind == "quantity":
            n = self._pending_quantity
            self._pending_quantity = None
            if not (1 <= n <= len(cands)):
                raise ValueError(f"quantity {n} not found among {len(cands)} candidates")
            return n - 1

        if slot_kind == "unit_op":
            entry = self._unit_entry_for_current()
            target_op = entry[0]
            target_item = entry[1] if len(entry) > 1 else None
            self._pending_quantity = entry[2] if len(entry) > 2 else 1
            for i, cand in enumerate(cands):
                if (
                    _op_name(cand, V.ACTION_FARMER_OP, C.FARMER_OP_NAMES) == target_op
                    and _item_name(cand) == target_item
                ):
                    return i
            raise ValueError(f"{entry} not found among {len(cands)} candidates")

        # slot_kind == "market_op"
        if self._market_cursor >= len(self._market):
            # もう注文しない(STOP候補は常に最後)。
            return len(cands) - 1
        entry = self._market[self._market_cursor]
        self._market_cursor += 1
        if entry == list(A.MARKET_WAIT_ACTION):
            return next(i for i, cand in enumerate(cands) if V.ACTION_MARKET_WAIT[0] in cand.index)
        target_op = entry[0]
        target_item = entry[1] if len(entry) > 1 else None
        self._pending_quantity = entry[2] if len(entry) > 2 else 1
        for i, cand in enumerate(cands):
            if any(
                special in cand.index
                for special in (V.ACTION_MARKET_WAIT[0], V.ACTION_MARKET_STOP[0])
            ):
                continue
            if (
                _op_name(cand, V.ACTION_MARKET_OP, C.MARKET_OP_NAMES) == target_op
                and _item_name(cand) == target_item
            ):
                return i
        raise ValueError(f"{entry} not found among {len(cands)} candidates")

    def _unit_entry_for_current(self):
        # choose()呼び出し順(farmer→hand0→hand1→...)に合わせて1つずつ消費する。
        if self._hand_cursor == 0:
            entry = self._farmer
        else:
            entry = self._hands[self._hand_cursor - 1]
        self._hand_cursor += 1
        return entry


class _NormalizingChooser:
    """保存済み行動を、無効なエントリを代替候補へ正規化しながらたどる。

    _TeacherForceChooserと同じ巡回順序・照合ロジックだが、候補に一致しない場合に
    ValueErrorを出す代わりに常に有効な代替候補を選ぶ(normalize_expert_action参照):
    unit行動はPASS、実行不可能な市場注文はMARKET_WAIT(STOPにすると後続の有効な
    注文まで失われるため)、実行可能上限を超える数量は上限にクランプする。
    PASS/MARKET_WAITは常に候補として存在するため、choose()は失敗しない。
    """

    def __init__(self, action: dict):
        self._farmer = list(action["farmer"])
        self._hands = [list(h) for h in action["hands"]]
        self._market = [list(m) for m in action["market"]]
        self._hand_cursor = 0
        self._market_cursor = 0
        self._pending_quantity: int | None = None

    def choose(self, cands: list, slot_kind: str) -> int:
        if slot_kind == "quantity":
            n = self._pending_quantity
            self._pending_quantity = None
            return min(n, len(cands)) - 1

        if slot_kind == "unit_op":
            entry = self._unit_entry_for_current()
            target_op = entry[0]
            target_item = entry[1] if len(entry) > 1 else None
            self._pending_quantity = entry[2] if len(entry) > 2 else 1
            for i, cand in enumerate(cands):
                if (
                    _op_name(cand, V.ACTION_FARMER_OP, C.FARMER_OP_NAMES) == target_op
                    and _item_name(cand) == target_item
                ):
                    return i
            self._pending_quantity = None
            return next(
                i
                for i, cand in enumerate(cands)
                if _op_name(cand, V.ACTION_FARMER_OP, C.FARMER_OP_NAMES) == "PASS"
            )

        # slot_kind == "market_op"
        if self._market_cursor >= len(self._market):
            return len(cands) - 1  # STOP
        entry = self._market[self._market_cursor]
        self._market_cursor += 1
        if entry == list(A.MARKET_WAIT_ACTION):
            return next(i for i, cand in enumerate(cands) if V.ACTION_MARKET_WAIT[0] in cand.index)
        target_op = entry[0]
        target_item = entry[1] if len(entry) > 1 else None
        self._pending_quantity = entry[2] if len(entry) > 2 else 1
        for i, cand in enumerate(cands):
            if any(
                special in cand.index
                for special in (V.ACTION_MARKET_WAIT[0], V.ACTION_MARKET_STOP[0])
            ):
                continue
            if (
                _op_name(cand, V.ACTION_MARKET_OP, C.MARKET_OP_NAMES) == target_op
                and _item_name(cand) == target_item
            ):
                return i
        self._pending_quantity = None
        return next(i for i, cand in enumerate(cands) if V.ACTION_MARKET_WAIT[0] in cand.index)

    def _unit_entry_for_current(self):
        if self._hand_cursor == 0:
            entry = self._farmer
        else:
            entry = self._hands[self._hand_cursor - 1]
        self._hand_cursor += 1
        return entry


def _canonicalize_expert_noops(obs: dict, action: dict) -> dict:
    """環境がno-opにする入力とatomic PLANT失敗を明示的な行動へ変換する。"""
    if not isinstance(action, dict):
        action = {}

    player = obs["player"]
    farm = obs["farms"][player]
    positions = [farm["farmer"], *farm["hands"]]

    farmer = action.get("farmer", ["PASS"])
    raw_hands = action.get("hands", [])
    if not isinstance(raw_hands, list):
        raw_hands = []
    unit_entries = [farmer, *raw_hands[: len(farm["hands"])]]
    unit_entries.extend([["PASS"]] * (len(positions) - len(unit_entries)))

    normalized_units = []
    for entry, pos in zip(unit_entries, positions, strict=True):
        if not isinstance(entry, list) or not entry:
            normalized_units.append(["PASS"])
            continue

        entry = list(entry)
        op_name = entry[0]
        item_name = entry[1] if len(entry) > 1 else None
        x, y = pos
        tile = farm["tiles"][y][x]
        if len(entry) >= 3 and DS.requires_quantity(op_name, item_name, tile):
            try:
                entry[2] = int(entry[2])
            except (TypeError, ValueError):
                entry = ["PASS"]
            else:
                if entry[2] <= 0:
                    entry = ["PASS"]
        normalized_units.append(entry)

    # 元環境は、作物ごとのPLANT要求が種数を超えるとその作物の要求を全て捨てる。
    demand: dict[str, int] = {}
    for entry in normalized_units:
        if len(entry) >= 2 and entry[0] == "PLANT":
            demand[entry[1]] = demand.get(entry[1], 0) + 1
    seeds = obs["private"]["seeds"]
    blocked = {crop for crop, n in demand.items() if n > seeds.get(crop, 0)}
    normalized_units = [
        ["PASS"] if len(entry) >= 2 and entry[0] == "PLANT" and entry[1] in blocked else entry
        for entry in normalized_units
    ]

    raw_market = action.get("market", [])
    if not isinstance(raw_market, list):
        raw_market = []
    market = []
    quantity_ops = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}
    for entry in raw_market:
        if not isinstance(entry, list) or not entry:
            market.append(list(A.MARKET_WAIT_ACTION))
            continue
        entry = list(entry)
        if entry[0] in quantity_ops:
            if len(entry) < 3:
                market.append(list(A.MARKET_WAIT_ACTION))
                continue
            try:
                entry[2] = int(entry[2])
            except (TypeError, ValueError):
                market.append(list(A.MARKET_WAIT_ACTION))
                continue
            if entry[2] <= 0:
                market.append(list(A.MARKET_WAIT_ACTION))
                continue
        market.append(entry)

    return {
        "farmer": normalized_units[0],
        "hands": normalized_units[1:],
        "market": market,
    }


@dataclass(slots=True)
class _DecodeEnv:
    farm: dict
    shed: dict
    seeds: dict
    market: dict
    inventories: list
    day: int
    turns_per_day: int
    shed_capacity: int
    hire_mult: float
    max_market_orders: int


def _decode_turn_gen(env: _DecodeEnv):
    """farmer→hands→市場注文をデコードする。

    Args:
        env: コピー済みの1環境分の状態と設定。

    Yields:
        候補、スロット種別、位置ID、盤面位置ID、unit情報。send()で候補indexを受け取る。

    Returns:
        kaggle-environments形式の1ターン分の行動。
    """
    farm = env.farm
    shed = env.shed
    seeds = env.seeds
    market = env.market
    inventories = env.inventories
    day = env.day
    turns_per_day = env.turns_per_day
    shed_capacity = env.shed_capacity
    hire_mult = env.hire_mult
    max_market_orders = env.max_market_orders

    def do_unit(pos, inventory, action_position, quantity_position):
        cands = A.legal_unit_actions(farm, shed, seeds, inventory, pos, day, shed_capacity)
        unit_context = tokenize.encode_unit_context(inventory)
        x, y = pos
        board_position_id = y * V.BOARD_SIZE + x
        idx = yield (cands, "unit_op", action_position, board_position_id, unit_context)
        chosen = cands[idx]
        op_name = _op_name(chosen, V.ACTION_FARMER_OP, C.FARMER_OP_NAMES)
        item_name = _item_name(chosen)
        tile = farm["tiles"][y][x]

        n = 1
        if DS.requires_quantity(op_name, item_name, tile):
            max_q = DS.max_executable_quantity(
                op_name, item_name, farm, shed, market, shed_capacity, inventory=inventory
            )
            qty_cands = A.quantity_candidates(max_q)
            qidx = yield (qty_cands, "quantity", quantity_position, L.NO_POSITION, None)
            n = qidx + 1

        DS.commit_unit_action(
            farm,
            shed,
            seeds,
            op_name,
            item_name,
            pos,
            day,
            n=n,
            shed_capacity=shed_capacity,
            inventory=inventory,
            turns_per_day=turns_per_day,
        )
        return _entry_from(op_name, item_name, n)

    farmer_pos = tuple(farm["farmer"])
    farmer_entry = yield from do_unit(
        farmer_pos, inventories[0], L.FARMER_POSITION, L.FARMER_QUANTITY_POSITION
    )

    hands_entries = []
    for h, hand_pos in enumerate(farm["hands"]):
        entry = yield from do_unit(
            tuple(hand_pos), inventories[h + 1], L.hand_position(h), L.hand_quantity_position(h)
        )
        hands_entries.append(entry)

    market_entries = []
    m = 0
    while True:
        cands = A.legal_market_actions(farm, shed, market, hire_mult, shed_capacity) + [
            A.market_wait_candidate(),
            A.market_stop_candidate(),
        ]
        idx = yield (cands, "market_op", L.market_position(m), L.NO_POSITION, None)
        chosen = cands[idx]
        if V.ACTION_MARKET_STOP[0] in chosen.index:
            break

        if V.ACTION_MARKET_WAIT[0] in chosen.index:
            market_entries.append(list(A.MARKET_WAIT_ACTION))
            m += 1
            if m >= max_market_orders:
                break
            continue

        op_name = _op_name(chosen, V.ACTION_MARKET_OP, C.MARKET_OP_NAMES)
        item_name = _item_name(chosen)
        n = 1
        if DS.requires_quantity(op_name, item_name):
            max_q = DS.max_executable_quantity(
                op_name, item_name, farm, shed, market, shed_capacity
            )
            qty_cands = A.quantity_candidates(max_q)
            qidx = yield (qty_cands, "quantity", L.market_quantity_position(m), L.NO_POSITION, None)
            n = qidx + 1
        DS.commit_market_action(farm, shed, market, op_name, item_name, n, shed_capacity, hire_mult)
        market_entries.append(_entry_from(op_name, item_name, n, force_quantity=True))
        m += 1
        if m >= max_market_orders:
            break

    return {"farmer": farmer_entry, "hands": hands_entries, "market": market_entries}


def _envs_from_observations(
    observations: list[dict],
    turns_per_day: int,
    shed_capacity: int,
    hire_mult: float,
    max_market_orders: int,
) -> list[_DecodeEnv]:
    """観測を破壊的に更新可能なデコード用状態へコピーする。

    汎用のcopy.deepcopyは使わず、decode_state.commit_unit_action/commit_market_action
    が実際に書き換える範囲だけを手動でコピーする(プロファイリングの結果、
    copy.deepcopyがBC学習の総実行時間の1〜2割を占めていた)。安全性の根拠:

    - farm["tiles"]: セル丸ごとの置換(PLANT/DIG等)とセル内フィールドの書き換え
      (WATER等)の両方が起きるため、行ごと・dict型セルごとに複製が必要
    - farm["farmer"]/["hands"]/["unlocked_quadrants"]: 参照を直接書き換えることは
      無く、常に新しいlistで再代入される(commit_market_action参照)ため、
      トップレベルの浅いコピーだけで安全(元のlistを共有していても再代入時に
      上書きされるだけで元は壊れない)
    - shed/seeds/market["inventory"]/market["prices"]: 全てキー単位の再代入のみ
      (ネストした書き換えは無い)ため、1階層の浅いコピーで十分
    - private["inventories"](各ユニットの持ち物): commit_unit_actionは読み取る
      だけで一切書き換えない(=deepcopyの再帰コピーは不要)が、呼び出し側が
      観測と参照を共有しないことに依存し得るため、各ユニット分の辞書だけは
      浅くコピーする

    Args:
        observations: ターン開始時点の観測。
        turns_per_day: 1日当たりのターン数。
        shed_capacity: 納屋容量。
        hire_mult: 雇用費倍率。
        max_market_orders: 1ターンの市場注文上限。

    Returns:
        観測と共有参照を持たないデコード用状態。

    Raises:
        ValueError: 市場注文上限が設定範囲外の場合。
    """
    # 0だと最初の注文が上限判定より先に通るため、設定契約をここで検証する。
    if not (1 <= max_market_orders <= C.MAX_MARKET_ORDERS):
        raise ValueError(
            f"max_market_orders must be in [1, {C.MAX_MARKET_ORDERS}], got {max_market_orders}"
        )
    envs = []
    for obs in observations:
        player = obs["player"]
        farm_src = obs["farms"][player]
        farm = {
            **farm_src,
            "tiles": [
                [dict(cell) if isinstance(cell, dict) else cell for cell in row]
                for row in farm_src["tiles"]
            ],
        }
        private = obs["private"]
        market_src = obs["market"]
        envs.append(
            _DecodeEnv(
                farm=farm,
                shed=dict(private["shed"]),
                seeds=dict(private["seeds"]),
                market={
                    "inventory": dict(market_src["inventory"]),
                    "prices": dict(market_src["prices"]),
                },
                inventories=[dict(inv) for inv in private["inventories"]],
                day=obs["day"],
                turns_per_day=turns_per_day,
                shed_capacity=shed_capacity,
                hire_mult=hire_mult,
                max_market_orders=max_market_orders,
            )
        )
    return envs


def normalize_expert_action(
    obs: dict,
    action: dict,
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1,
    max_market_orders: int = C.MAX_MARKET_ORDERS,
) -> dict:
    """生のexpert行動を、この観測時点で方策が表現できる形へ正規化する。

    生リプレイには、シミュレータ側で黙ってno-opまたはクランプされる無効な注文が
    含まれる(README.mdの「BC用リプレイの入力契約」参照)。以下の規則で
    正規化する:
      - 候補に無いunit行動 → PASS
      - 候補に無い市場注文 → MARKET_WAIT(STOPにすると後続の有効な注文まで
        失われるため、キュー位置だけ消費して次のスロットへ進む)
      - 数量0以下のunit行動 → PASS、数量不正の市場注文 → MARKET_WAIT
      - 同一作物のPLANT要求が種数を超える場合 → その作物の全PLANTをPASS
      - 実行可能上限を超える正の数量 → 上限にクランプ
    PASS/MARKET_WAITは常にその時点の候補として存在するため、この関数自体は
    ValueErrorを出さない。ただし正規化ロジックとvalidate_policy_actionの定義が
    ずれていないか確認するため、呼び出し側で返り値をvalidate_policy_actionに
    通すことを推奨する。

    Args:
        obs: player視点の観測。
        action: 正規化対象のexpert行動。
        turns_per_day: 1日当たりのターン数。
        shed_capacity: 納屋容量。
        hire_mult: 雇用費倍率。
        max_market_orders: 1ターンの市場注文上限。

    Returns:
        dict: kaggle-environments形式の正規化済みaction。
    """
    env = _envs_from_observations(
        [obs], turns_per_day, shed_capacity, hire_mult, max_market_orders
    )[0]
    gen = _decode_turn_gen(env)
    chooser = _NormalizingChooser(_canonicalize_expert_noops(obs, action))
    try:
        step = next(gen)
        while True:
            cands, slot_kind, *_ = step
            idx = chooser.choose(cands, slot_kind)
            step = gen.send(idx)
    except StopIteration as e:
        return e.value
