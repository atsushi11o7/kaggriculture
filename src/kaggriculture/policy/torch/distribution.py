"""ネットワークとPPOのロールアウト・更新処理をつなぐ行動分布。

サンプリングと教師強制は共通の自己回帰ロジックを使い、決定順・合法候補・仮状態
更新を一致させる。PPOの再評価・rollout・推論ではdropoutを無効化し、BC用APIは
呼び出し側が設定したtrain/eval modeを維持する。
"""

import contextlib
from dataclasses import dataclass

import torch

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.torch import actions as A
from kaggriculture.policy.torch import decode as DS
from kaggriculture.policy.torch import features as F
from kaggriculture.policy.torch import model as M
from kaggriculture.policy.torch import tokenize
from kaggriculture.rules import constants as C


@contextlib.contextmanager
def _eval_mode(net: M.PolicyValueNet):
    """一時的にdropoutを無効化し、呼び出し前の.training状態へ復元する。"""
    was_training = net.training
    net.eval()
    try:
        yield
    finally:
        net.train(was_training)


def _net_device(net: M.PolicyValueNet) -> torch.device:
    """モデル入力を置くデバイスを返す。"""
    return net.token_embedding.bag.weight.device


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


def _masked_entropy(
    scores: torch.Tensor, log_probs: torch.Tensor, probs: torch.Tensor
) -> torch.Tensor:
    """paddingを除いたentropyを計算する。

    Args:
        scores: paddingが-infの候補logit。
        log_probs: 候補の対数確率。
        probs: 候補の確率。

    Returns:
        行ごとのentropy。
    """
    safe_log_probs = log_probs.masked_fill(~torch.isfinite(scores), 0.0)
    return -(probs * safe_log_probs).sum(dim=-1)


def _padding_group_key(slot_kind: str, n_candidates: int) -> tuple:
    """候補数のグループ化キーを返す。

    Args:
        slot_kind: unit_op、market_op、quantityのいずれか。
        n_candidates: 候補数。

    Returns:
        数量候補を2の累乗幅に分けたキー。
    """
    if slot_kind != "quantity":
        return (slot_kind,)
    bucket = 8
    while n_candidates > bucket:
        bucket *= 2
    return (slot_kind, bucket)


def _score_candidates_grouped(
    net: M.PolicyValueNet,
    hidden: torch.Tensor,
    cands_per_row: list[list[V.SparseVector]],
    slot_kinds: list[str],
) -> torch.Tensor:
    """候補数の近い行をまとめて採点する。

    Args:
        net: 方策ネットワーク。
        hidden: スロットごとの隠れ状態。
        cands_per_row: 行ごとの合法候補。
        slot_kinds: 行ごとのスロット種別。

    Returns:
        padding列を-infにした候補logit。
    """
    n = hidden.shape[0]
    max_cands = max(len(c) for c in cands_per_row)
    scores = torch.full((n, max_cands), float("-inf"), device=hidden.device)
    groups: dict[tuple, list[int]] = {}
    for i, (kind, cands) in enumerate(zip(slot_kinds, cands_per_row, strict=True)):
        groups.setdefault(_padding_group_key(kind, len(cands)), []).append(i)
    for idxs in groups.values():
        group_scores = net.decoder.score_candidates_batch(
            hidden[idxs], [cands_per_row[i] for i in idxs]
        )
        scores[idxs, : group_scores.shape[1]] = group_scores
    return scores


def _pad_and_collate(
    tokens_per_env: list[list[V.SparseVector]],
    position_ids_per_env: list[list[int]],
    board_position_ids_per_env: list[list[int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """環境ごとの可変長系列をpaddingしてcollateする。

    Args:
        tokens_per_env: 環境ごとの決定トークン列。
        position_ids_per_env: 固定スロット位置ID列。
        board_position_ids_per_env: unitの盤面位置ID列。
        device: 出力先デバイス。

    Returns:
        collate済み疎特徴量、位置ID、盤面位置ID、padding mask。
    """
    max_len = max(len(seq) for seq in tokens_per_env)
    flat_tokens: list[V.SparseVector] = []
    batch_pid, batch_bpid, batch_padmask = [], [], []
    for tokens, pids, bpids in zip(
        tokens_per_env, position_ids_per_env, board_position_ids_per_env, strict=True
    ):
        pad_n = max_len - len(tokens)
        flat_tokens.extend(tokens)
        flat_tokens.extend(V.SparseVector() for _ in range(pad_n))
        batch_pid.append(pids + [0] * pad_n)
        batch_bpid.append(bpids + [L.NO_POSITION] * pad_n)
        batch_padmask.append([False] * len(tokens) + [True] * pad_n)

    d_index, d_value, d_offset = F.collate(flat_tokens, device=device)
    pid_t = torch.tensor(batch_pid, dtype=torch.long, device=device)
    bpid_t = torch.tensor(batch_bpid, dtype=torch.long, device=device)
    padmask_t = torch.tensor(batch_padmask, dtype=torch.bool, device=device)
    return d_index, d_value, d_offset, pid_t, bpid_t, padmask_t


def _decode_turn_batch(
    net: M.PolicyValueNet, memory_batch: torch.Tensor, envs: list[_DecodeEnv]
) -> tuple[list[dict], torch.Tensor, torch.Tensor, list[int]]:
    """複数環境を決定深度ごとにまとめてサンプリングする。

    Args:
        net: 方策ネットワーク。
        memory_batch: 各環境の公開Encoder出力。
        envs: デコード用状態。

    Returns:
        行動、log probability、entropy、決定数。
    """
    device = memory_batch.device
    n_envs = len(envs)
    gens = [_decode_turn_gen(env) for env in envs]

    tokens_per_env: list[list[V.SparseVector]] = [[] for _ in range(n_envs)]
    position_ids_per_env: list[list[int]] = [[] for _ in range(n_envs)]
    board_position_ids_per_env: list[list[int]] = [[] for _ in range(n_envs)]
    next_token_per_env = [V.SparseVector() for _ in range(n_envs)]

    total_log_prob = torch.zeros(n_envs, device=device)
    total_entropy = torch.zeros(n_envs, device=device)
    num_decisions = [0] * n_envs
    results: list = [None] * n_envs

    pending: dict[int, tuple] = {}
    active: list[int] = []
    for i in range(n_envs):
        try:
            pending[i] = next(gens[i])
            active.append(i)
        except StopIteration as e:
            results[i] = e.value

    while active:
        for i in active:
            cands, _, position_id, board_position_id, unit_context = pending[i]
            nt = next_token_per_env[i]
            if unit_context is not None:
                nt.extend(unit_context)
            tokens_per_env[i].append(nt)
            position_ids_per_env[i].append(position_id)
            board_position_ids_per_env[i].append(board_position_id)
            next_token_per_env[i] = V.SparseVector()

        d_index, d_value, d_offset, pid_t, bpid_t, padmask_t = _pad_and_collate(
            [tokens_per_env[i] for i in active],
            [position_ids_per_env[i] for i in active],
            [board_position_ids_per_env[i] for i in active],
            device,
        )
        memory_active = memory_batch[active]

        decoder_out = net.decoder(
            memory_active, d_index, d_value, d_offset, pid_t, bpid_t, padmask_t
        )

        hidden = torch.stack(
            [decoder_out[row, len(tokens_per_env[i]) - 1] for row, i in enumerate(active)]
        )
        cands_per_row = [pending[i][0] for i in active]
        slot_kinds = [pending[i][1] for i in active]
        scores = _score_candidates_grouped(net, hidden, cands_per_row, slot_kinds)
        log_probs = torch.log_softmax(scores, dim=-1)
        probs = log_probs.exp()
        row_entropy = _masked_entropy(scores, log_probs, probs)
        # padding列の確率は0なので選ばれない。
        idx_batch = torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()

        next_active = []
        for row, i in enumerate(active):
            idx = idx_batch[row]
            cands, _, _, _, _ = pending[i]
            total_log_prob[i] = total_log_prob[i] + log_probs[row, idx]
            total_entropy[i] = total_entropy[i] + row_entropy[row]
            num_decisions[i] += 1

            next_token_per_env[i].extend(cands[idx])

            try:
                pending[i] = gens[i].send(idx)
                next_active.append(i)
            except StopIteration as e:
                results[i] = e.value
        active = next_active

    return results, total_log_prob, total_entropy, num_decisions


def _teacher_force_trace(
    env: _DecodeEnv, action: dict
) -> tuple[list[V.SparseVector], list[int], list[int], list[tuple[list, str, int]]]:
    """ネットワークを使わず、1環境分の教師強制列を再構築する。

    _decode_turn_gen()を_TeacherForceChooserだけで完走させ、決定トークン列と
    各スロットの(候補, スロット種別, 選択index)を求める。Encoder/Decoderも
    テンソルも一切使わない、合法候補列挙とdecode_stateの仮状態更新だけの処理。

    Returns:
        (tokens, position_ids, board_position_ids, slots)。

    Raises:
        ValueError: actionのどれかのエントリが、その時点の合法候補に無い場合
        (_TeacherForceChooser.choose参照)。
    """
    gen = _decode_turn_gen(env)
    chooser = _TeacherForceChooser(action)
    tokens: list[V.SparseVector] = []
    position_ids: list[int] = []
    board_position_ids: list[int] = []
    slots: list[tuple[list, str, int]] = []
    next_token = V.SparseVector()

    try:
        step = next(gen)
        while True:
            cands, slot_kind, position_id, board_position_id, unit_context = step
            if unit_context is not None:
                next_token.extend(unit_context)
            tokens.append(next_token)
            position_ids.append(position_id)
            board_position_ids.append(board_position_id)
            next_token = V.SparseVector()

            idx = chooser.choose(cands, slot_kind)
            slots.append((cands, slot_kind, idx))
            next_token.extend(cands[idx])

            step = gen.send(idx)
    except StopIteration:
        pass

    return tokens, position_ids, board_position_ids, slots


def _decode_turn_teacher_force(
    net: M.PolicyValueNet, memory_batch: torch.Tensor, envs: list[_DecodeEnv], actions: list[dict]
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """保存済み行動を教師強制でまとめて評価する。

    選択列を先に再構築し、Decoderを1回だけ実行するため計算量はO(T^2)。

    Args:
        net: 方策ネットワーク。
        memory_batch: 各環境の公開Encoder出力。
        envs: デコード用状態。
        actions: 保存済み行動。

    Returns:
        log probability、entropy、決定数。
    """
    device = memory_batch.device
    n_envs = len(envs)

    # フェーズ1: ネットワークを使わず、全環境・全スロットの候補とindexを求める。
    tokens_per_env: list[list[V.SparseVector]] = []
    position_ids_per_env: list[list[int]] = []
    board_position_ids_per_env: list[list[int]] = []
    slots_per_env: list[list[tuple[list, str, int]]] = []  # (candidates, slot_kind, chosen_idx)

    for env, action in zip(envs, actions, strict=True):
        tokens, position_ids, board_position_ids, slots = _teacher_force_trace(env, action)
        tokens_per_env.append(tokens)
        position_ids_per_env.append(position_ids)
        board_position_ids_per_env.append(board_position_ids)
        slots_per_env.append(slots)

    # フェーズ2: 環境ごとの全系列をpaddingで揃え、1回だけnet.decoder()を呼ぶ。
    d_index, d_value, d_offset, pid_t, bpid_t, padmask_t = _pad_and_collate(
        tokens_per_env, position_ids_per_env, board_position_ids_per_env, device
    )
    decoder_out = net.decoder(memory_batch, d_index, d_value, d_offset, pid_t, bpid_t, padmask_t)

    # フェーズ3: 全環境・全スロットの候補スコアを1回のバッチ処理でまとめて計算する
    # (スロットごとにPythonループでscore_candidatesを呼ぶと、決定数分のGPU呼び出しが
    # 積み重なりPPO更新のたびに支配的コストになるため)。
    flat_hidden = []
    flat_cands = []
    flat_slot_kinds = []
    for i, slots in enumerate(slots_per_env):
        for slot_i, (cands, slot_kind, _) in enumerate(slots):
            flat_hidden.append(decoder_out[i, slot_i])
            flat_cands.append(cands)
            flat_slot_kinds.append(slot_kind)

    scores = _score_candidates_grouped(net, torch.stack(flat_hidden), flat_cands, flat_slot_kinds)
    log_probs = torch.log_softmax(scores, dim=-1)
    probs = log_probs.exp()
    entropy_per_slot = _masked_entropy(scores, log_probs, probs)

    total_log_prob = torch.zeros(n_envs, device=device)
    total_entropy = torch.zeros(n_envs, device=device)
    num_decisions = [len(slots) for slots in slots_per_env]
    flat_i = 0
    for i, slots in enumerate(slots_per_env):
        for _, _, idx in slots:
            total_log_prob[i] = total_log_prob[i] + log_probs[flat_i, idx]
            total_entropy[i] = total_entropy[i] + entropy_per_slot[flat_i]
            flat_i += 1

    return total_log_prob, total_entropy, num_decisions


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


def _encode_observations(
    net: M.PolicyValueNet,
    observations: list[dict],
    turns_per_day: int,
    counters: list[dict | None] | None = None,
) -> torch.Tensor:
    """公開観測をまとめて符号化する。

    Args:
        net: 方策ネットワーク。
        observations: player視点の観測。
        turns_per_day: 1日当たりのターン数。
        counters: 観測ごとの累積実績。

    Returns:
        公開Encoder出力。

    Raises:
        ValueError: 空batch、長さ不一致、またはモデル設定と履歴入力が合わない場合。
    """
    if not observations:
        raise ValueError("observations must not be empty")
    n = len(observations)
    if counters is not None and len(counters) != n:
        raise ValueError("counters length must match observations")

    device = _net_device(net)

    if net.uses_episode_history:
        if counters is None or any(c is None for c in counters):
            raise ValueError(
                "net.uses_episode_history=True requires counters for every observation "
                "(pass {} explicitly for a fresh episode, not None)"
            )
    elif counters is not None:
        raise ValueError("net.uses_episode_history=False but counters was given")
    counters_list = counters if counters is not None else [None] * n

    all_tokens = []
    for obs, c in zip(observations, counters_list, strict=True):
        all_tokens.extend(tokenize.get_encoder_input(obs, turns_per_day, c))
    index, value_t, offset = F.collate(all_tokens, device=device)
    return net.encode(index, value_t, offset)


def _evaluate_values(
    net: M.PolicyValueNet,
    memory: torch.Tensor,
    observations: list[dict],
    opponent_privates: list[dict | None] | None,
) -> torch.Tensor:
    """公開表現と、設定時のみ非公開表現から価値を計算する。

    Args:
        net: 方策ネットワーク。
        memory: 公開Encoder出力。
        observations: player視点の観測。
        opponent_privates: 観測ごとの相手private。

    Returns:
        形状(batch,)の価値。

    Raises:
        ValueError: critic設定とprivate入力が合わない場合。
    """
    if opponent_privates is not None and len(opponent_privates) != len(observations):
        raise ValueError("opponent_privates length must match observations")
    if not net.uses_asymmetric_critic:
        if opponent_privates is not None:
            raise ValueError("symmetric critic does not accept opponent_privates")
        return net.value(memory).squeeze(-1)

    if opponent_privates is None or any(private is None for private in opponent_privates):
        raise ValueError("asymmetric critic requires opponent_privates for every observation")

    device = _net_device(net)
    tokens = []
    position_ids = []
    padding_masks = []
    for obs, private in zip(observations, opponent_privates, strict=True):
        obs_tokens, positions, padding = tokenize.get_privileged_critic_input(obs, private)
        tokens.extend(obs_tokens)
        position_ids.append(positions)
        padding_masks.append(padding)

    index, value, offset = F.collate(tokens, device=device)
    positions = torch.tensor(position_ids, dtype=torch.long, device=device)
    padding = torch.tensor(padding_masks, dtype=torch.bool, device=device)
    privileged_memory = net.encode_privileged(index, value, offset, positions, padding)
    return net.value(memory, privileged_memory).squeeze(-1)


def act_batch(
    net: M.PolicyValueNet,
    observations: list[dict],
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1,
    max_market_orders: int = C.MAX_MARKET_ORDERS,
    counters: list[dict | None] | None = None,
    opponent_privates: list[dict | None] | None = None,
) -> tuple[list[dict], torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """PPO rollout用に複数局面の行動と価値をまとめて計算する。

    Args:
        net: 方策・価値ネットワーク。
        observations: player視点の観測batch。
        turns_per_day: 1日当たりのターン数。
        shed_capacity: 納屋容量。
        hire_mult: 雇用費倍率。
        max_market_orders: 1ターンの市場注文上限。
        counters: 履歴を使う場合の累積実績。
        opponent_privates: 非対称critic用の相手private。

    Returns:
        行動、価値、log probability、entropy、決定数。Tensorは形状(batch,)。
    """
    with _eval_mode(net), torch.no_grad():
        memory = _encode_observations(net, observations, turns_per_day, counters)
        values = _evaluate_values(net, memory, observations, opponent_privates)
        envs = _envs_from_observations(
            observations, turns_per_day, shed_capacity, hire_mult, max_market_orders
        )
        actions, log_probs, entropies, num_decisions = _decode_turn_batch(net, memory, envs)
    return actions, values, log_probs, entropies, num_decisions


def _evaluate_policy_batch(
    net: M.PolicyValueNet,
    obs_action_pairs: list[tuple[dict, dict]],
    turns_per_day: int,
    shed_capacity: int,
    hire_mult: float,
    max_market_orders: int,
    counters: list[dict | None] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """教師強制に共通するEncoder・Decoder処理を実行する。"""
    observations = [obs for obs, _ in obs_action_pairs]
    actions = [action for _, action in obs_action_pairs]
    memory = _encode_observations(net, observations, turns_per_day, counters)
    envs = _envs_from_observations(
        observations, turns_per_day, shed_capacity, hire_mult, max_market_orders
    )
    log_probs, entropies, num_decisions = _decode_turn_teacher_force(net, memory, envs, actions)
    return memory, log_probs, entropies, num_decisions


def evaluate_policy_batch(
    net: M.PolicyValueNet,
    obs_action_pairs: list[tuple[dict, dict]],
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1,
    max_market_orders: int = C.MAX_MARKET_ORDERS,
    counters: list[dict | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """保存済み行動をBC用にactorだけで教師強制評価する。

    netのtrain/eval modeを変更しないため、BCでdropoutを使うかは呼び出し側が決める。

    Args:
        net: 方策ネットワーク。
        obs_action_pairs: 観測とexpert行動。
        turns_per_day: 1日当たりのターン数。
        shed_capacity: 納屋容量。
        hire_mult: 雇用費倍率。
        max_market_orders: 1ターンの市場注文上限。
        counters: 各観測時点の累積実績。

    Returns:
        行動全体のlog probability、entropy、決定数。

    Raises:
        ValueError: expert行動を合法候補として再構築できない場合。
    """
    _, log_probs, entropies, num_decisions = _evaluate_policy_batch(
        net,
        obs_action_pairs,
        turns_per_day,
        shed_capacity,
        hire_mult,
        max_market_orders,
        counters,
    )
    return log_probs, entropies, num_decisions


def evaluate_actions_batch(
    net: M.PolicyValueNet,
    obs_action_pairs: list[tuple[dict, dict]],
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1,
    max_market_orders: int = C.MAX_MARKET_ORDERS,
    counters: list[dict | None] | None = None,
    opponent_privates: list[dict | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """保存済み行動をPPO更新用にまとめて再評価する。

    Args:
        net: 方策・価値ネットワーク。
        obs_action_pairs: ロールアウト時の観測と行動。
        turns_per_day: 1日当たりのターン数。
        shed_capacity: 納屋容量。
        hire_mult: 雇用費倍率。
        max_market_orders: 1ターンの市場注文上限。
        counters: ロールアウト時の累積実績。
        opponent_privates: ロールアウト時の相手private。

    Returns:
        価値、log probability、entropy、決定数。

    Raises:
        ValueError: 保存済み行動を合法候補として再構築できない場合。
    """
    observations = [obs for obs, _ in obs_action_pairs]
    with _eval_mode(net):
        memory, log_probs, entropies, num_decisions = _evaluate_policy_batch(
            net,
            obs_action_pairs,
            turns_per_day,
            shed_capacity,
            hire_mult,
            max_market_orders,
            counters,
        )
        values = _evaluate_values(net, memory, observations, opponent_privates)
    return values, log_probs, entropies, num_decisions


def act(
    net: M.PolicyValueNet,
    obs: dict,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1,
    max_market_orders: int = C.MAX_MARKET_ORDERS,
    counters: dict | None = None,
    opponent_private: dict | None = None,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """単一局面の行動と価値をPPO rollout用に計算する。

    Args:
        net: 方策・価値ネットワーク。
        obs: player視点の観測。
        turns_per_day: 1日当たりのターン数。
        shed_capacity: 納屋容量。
        hire_mult: 雇用費倍率。
        max_market_orders: 1ターンの市場注文上限。
        counters: 履歴を使う場合の累積実績。
        opponent_private: 非対称critic用の相手private。

    Returns:
        行動、価値、log probability、entropy、決定数。
    """
    actions, values, log_probs, entropies, num_decisions = act_batch(
        net,
        [obs],
        turns_per_day,
        shed_capacity,
        hire_mult,
        max_market_orders,
        [counters] if counters is not None else None,
        [opponent_private] if opponent_private is not None else None,
    )
    return actions[0], values[0], log_probs[0], entropies[0], num_decisions[0]


def predict_actions_batch(
    net: M.PolicyValueNet,
    observations: list[dict],
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1,
    max_market_orders: int = C.MAX_MARKET_ORDERS,
    counters: list[dict | None] | None = None,
) -> list[dict]:
    """criticを実行せず、提出・評価用の行動batchを生成する。

    Args:
        net: 方策ネットワーク。
        observations: player視点の観測batch。
        turns_per_day: 1日当たりのターン数。
        shed_capacity: 納屋容量。
        hire_mult: 雇用費倍率。
        max_market_orders: 1ターンの市場注文上限。
        counters: 履歴を使う場合の累積実績。

    Returns:
        生成した行動。
    """
    with _eval_mode(net), torch.no_grad():
        memory = _encode_observations(net, observations, turns_per_day, counters)
        envs = _envs_from_observations(
            observations,
            turns_per_day,
            shed_capacity,
            hire_mult,
            max_market_orders,
        )
        actions, _, _, _ = _decode_turn_batch(net, memory, envs)
    return actions


def predict_action(
    net: M.PolicyValueNet,
    obs: dict,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1,
    max_market_orders: int = C.MAX_MARKET_ORDERS,
    counters: dict | None = None,
) -> dict:
    """criticを実行せず、提出・評価用の単一行動を生成する。

    Args:
        net: 方策ネットワーク。
        obs: player視点の観測。
        turns_per_day: 1日当たりのターン数。
        shed_capacity: 納屋容量。
        hire_mult: 雇用費倍率。
        max_market_orders: 1ターンの市場注文上限。
        counters: 履歴を使う場合の累積実績。

    Returns:
        生成した行動。
    """
    return predict_actions_batch(
        net,
        [obs],
        turns_per_day=turns_per_day,
        shed_capacity=shed_capacity,
        hire_mult=hire_mult,
        max_market_orders=max_market_orders,
        counters=[counters] if counters is not None else None,
    )[0]


def evaluate_policy(
    net: M.PolicyValueNet,
    obs: dict,
    action: dict,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1,
    max_market_orders: int = C.MAX_MARKET_ORDERS,
    counters: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """単一のexpert行動をBC用にactorだけで教師強制評価する。

    Args:
        net: 方策ネットワーク。
        obs: expert行動直前の観測。
        action: expert行動。
        turns_per_day: 1日当たりのターン数。
        shed_capacity: 納屋容量。
        hire_mult: 雇用費倍率。
        max_market_orders: 1ターンの市場注文上限。
        counters: 観測時点の累積実績。

    Returns:
        行動全体のlog probability、entropy、決定数。
    """
    log_probs, entropies, num_decisions = evaluate_policy_batch(
        net,
        [(obs, action)],
        turns_per_day,
        shed_capacity,
        hire_mult,
        max_market_orders,
        [counters] if counters is not None else None,
    )
    return log_probs[0], entropies[0], num_decisions[0]


def validate_policy_action(
    obs: dict,
    action: dict,
    *,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1,
    max_market_orders: int = C.MAX_MARKET_ORDERS,
) -> None:
    """actionが、この観測時点の合法候補列から教師強制で再構築できるか検証する。

    Encoder/Decoderもテンソルも一切使わない(_teacher_force_trace参照。合法候補
    列挙とdecode_stateの仮状態更新だけを行う)ため、evaluate_policy/
    evaluate_actionsよりずっと安価。BCデータセット構築時、バッチ化・モデル実行の
    前に無効なexpert行動(README.mdの「BC用リプレイの入力契約」参照)を弾くのに使う。

    evaluate_policy_batch/evaluate_actions_batch自体は無効なactionを黙って
    スキップしたりはしない(事前検証済みのはずのバッチに無効な行動が混入した場合、
    それはデータパイプライン側のバグなので、従来通りValueErrorのままにする)。

    Args:
        obs: player視点の観測。
        action: 検証するaction。
        turns_per_day: 1日当たりのターン数。
        shed_capacity: 納屋容量。
        hire_mult: 雇用費倍率。
        max_market_orders: 1ターンの市場注文上限。

    Raises:
        ValueError: actionのどれかのエントリが、その時点の合法候補に無い場合
        (_TeacherForceChooser.choose参照。理由はメッセージに含まれる)。
    """
    env = _envs_from_observations(
        [obs], turns_per_day, shed_capacity, hire_mult, max_market_orders
    )[0]
    _teacher_force_trace(env, action)


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


def evaluate_actions(
    net: M.PolicyValueNet,
    obs: dict,
    action: dict,
    turns_per_day: int = 24,
    shed_capacity: int = 100,
    hire_mult: float = 1,
    max_market_orders: int = C.MAX_MARKET_ORDERS,
    counters: dict | None = None,
    opponent_private: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """単一の保存済み行動を現在のパラメータで再評価する。

    Args:
        net: 方策・価値ネットワーク。
        obs: ロールアウト時の観測。
        action: 保存済み行動。
        turns_per_day: 1日当たりのターン数。
        shed_capacity: 納屋容量。
        hire_mult: 雇用費倍率。
        max_market_orders: 1ターンの市場注文上限。
        counters: ロールアウト時の累積実績。
        opponent_private: ロールアウト時の相手private。

    Returns:
        価値、log probability、entropy、決定数。

    Raises:
        ValueError: 行動を合法候補として再構築できない場合。
    """
    values, log_probs, entropies, num_decisions = evaluate_actions_batch(
        net,
        [(obs, action)],
        turns_per_day,
        shed_capacity,
        hire_mult,
        max_market_orders,
        [counters] if counters is not None else None,
        [opponent_private] if opponent_private is not None else None,
    )
    return values[0], log_probs[0], entropies[0], num_decisions[0]
