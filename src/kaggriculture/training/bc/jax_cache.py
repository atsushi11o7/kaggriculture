"""Kaggle replayをJAX BC向け固定shape shardへ変換する。"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common import vocab as V
from kaggriculture.rules import constants as C
from kaggriculture.simulator.state import State

_CACHE_VERSION = 1
_STATE_PREFIX = "state__"
_ITEM_UNIT_OPS = {
    C.FARMER_OP_PICKUP,
    C.FARMER_OP_PLANT,
    C.FARMER_OP_PLACE,
}


def _unit_choice_table() -> dict[tuple[int, int], int]:
    entries = [(op, -1) for op in range(C.N_FARMER_OPS) if op not in _ITEM_UNIT_OPS]
    entries.extend((C.FARMER_OP_PLANT, item) for item in range(C.N_CROPS))
    entries.extend((C.FARMER_OP_PLACE, item) for item in range(C.N_SHED_ITEMS))
    entries.extend((C.FARMER_OP_PICKUP, item) for item in range(C.N_SHED_ITEMS))
    return {entry: i for i, entry in enumerate(entries)}


def _market_choice_table() -> dict[tuple[int, int], int]:
    entries = [(C.MARKET_OP_HIRE, -1), (C.MARKET_OP_BUY_LAND, -1)]
    entries.extend((C.MARKET_OP_BUY_SEED, item) for item in range(C.N_CROPS))
    entries.extend((C.MARKET_OP_BUY_ANIMAL, item) for item in range(C.N_ANIMALS))
    entries.extend(
        (C.MARKET_OP_BUY_PRODUCT, C.PRODUCTS.index(item)) for item in ("WHEAT", "FERTILIZER")
    )
    entries.extend((C.MARKET_OP_SELL, item) for item in range(C.N_PRODUCTS))
    entries.extend(((C.N_MARKET_OPS, -1), (C.N_MARKET_OPS + 1, -1)))
    return {entry: i for i, entry in enumerate(entries)}


_UNIT_CHOICE = _unit_choice_table()
_MARKET_CHOICE = _market_choice_table()


@dataclass(frozen=True)
class CacheRules:
    """expert正規化と候補列挙へ渡すゲーム規則。"""

    turns_per_day: int = 24
    shed_capacity: int = 100
    hire_mult: float = 1.0
    max_market_orders: int = C.MAX_MARKET_ORDERS
    min_player_reward: float | None = None


class BCBatch(NamedTuple):
    """JAX BCへ渡す固定shape batch。"""

    states: State
    players: np.ndarray
    choices: np.ndarray
    decision_mask: np.ndarray


def _dict_vector(values: dict, names: Sequence[str], dtype=np.int32) -> np.ndarray:
    return np.asarray([values.get(name, 0) for name in names], dtype=dtype)


def _parse_tile(tile) -> tuple:
    if tile is None:
        return (C.TILE_EMPTY, -1, 0, False, 0, 0, -1, -1, False, False, 0)
    if tile == "LOCKED":
        return (C.TILE_LOCKED, -1, 0, False, 0, 0, -1, -1, False, False, 0)
    kind = tile["kind"]
    if kind == "WEED":
        return (C.TILE_WEED, -1, 0, False, 0, 0, -1, -1, False, False, 0)
    if kind == "PLANT":
        return (
            C.TILE_PLANT,
            C.CROPS.index(tile["crop"]),
            tile["planted_day"],
            tile["watered_today"],
            tile["consecutive_unwatered"],
            tile["yield_units"],
            tile["max_lifespan_step"],
            tile["fertilized_until_day"],
            False,
            False,
            0,
        )
    tile_kind = C.TILE_COOP if kind == "COOP" else C.TILE_PASTURE
    if "animal" not in tile:
        return (tile_kind, -1, 0, False, 0, 0, -1, -1, False, False, 0)
    return (
        tile_kind,
        C.ANIMALS.index(tile["animal"]),
        tile["placed_day"],
        tile["fed_today"],
        tile["consecutive_unfed"],
        tile["yield_units"],
        -1,
        -1,
        tile["cared_today"],
        tile["fertilizer_available"],
        tile["pending_care_bonus"],
    )


def _empty_private() -> dict:
    return {
        "shed": {},
        "seeds": {},
        "inventories": [],
    }


def _parse_farm(farm: dict, private: dict, board_size: int) -> dict[str, np.ndarray]:
    if len(farm["hands"]) > C.MAX_HANDS:
        raise ValueError(f"replay has {len(farm['hands'])} hands; MAX_HANDS={C.MAX_HANDS}")
    tiles = [
        [_parse_tile(farm["tiles"][y][x]) for x in range(board_size)] for y in range(board_size)
    ]
    channels = tuple(np.asarray([[cell[i] for cell in row] for row in tiles]) for i in range(11))
    hands_pos = np.zeros((C.MAX_HANDS, 2), dtype=np.int32)
    hands_active = np.zeros(C.MAX_HANDS, dtype=np.bool_)
    for i, position in enumerate(farm["hands"]):
        hands_pos[i] = position
        hands_active[i] = True

    inventories = private.get("inventories", [])
    farmer_inventory = _dict_vector(inventories[0] if inventories else {}, C.SHED_ITEMS)
    hands_inventory = np.zeros((C.MAX_HANDS, C.N_SHED_ITEMS), dtype=np.int32)
    for i, inventory in enumerate(inventories[1 : C.MAX_HANDS + 1]):
        hands_inventory[i] = _dict_vector(inventory, C.SHED_ITEMS)

    return {
        "tiles_kind": channels[0].astype(np.int32),
        "tiles_crop_or_animal": channels[1].astype(np.int32),
        "tiles_planted_or_placed_day": channels[2].astype(np.int32),
        "tiles_watered_or_fed_today": channels[3].astype(np.bool_),
        "tiles_consecutive_unwatered_or_unfed": channels[4].astype(np.int32),
        "tiles_yield_units": channels[5].astype(np.int32),
        "tiles_max_lifespan_step": channels[6].astype(np.int32),
        "tiles_fertilized_until_day": channels[7].astype(np.int32),
        "tiles_cared_today": channels[8].astype(np.bool_),
        "tiles_fertilizer_available": channels[9].astype(np.bool_),
        "tiles_pending_care_bonus": channels[10].astype(np.int32),
        "farmer_pos": np.asarray(farm["farmer"], dtype=np.int32),
        "hands_pos": hands_pos,
        "hands_active": hands_active,
        "hires_today": np.asarray(farm["hires_today"], dtype=np.int32),
        "money": np.asarray(farm["money"], dtype=np.float32),
        "unlocked_quadrants": np.asarray(
            [name in farm["unlocked_quadrants"] for name in C.QUADRANTS], dtype=np.bool_
        ),
        "shed": _dict_vector(private.get("shed", {}), C.SHED_ITEMS),
        "seeds": _dict_vector(private.get("seeds", {}), C.CROPS),
        "farmer_inventory": farmer_inventory,
        "hands_inventory": hands_inventory,
    }


def observation_to_state(
    obs: dict, board_size: int = V.BOARD_SIZE, turns_per_day: int = 24
) -> State:
    """1人称observationをActor入力に十分な固定shape Stateへ変換する。"""
    player = int(obs["player"])
    farms = []
    for p in range(2):
        private = obs["private"] if p == player else _empty_private()
        farms.append(_parse_farm(obs["farms"][p], private, board_size))
    stacked = {name: np.stack([farms[0][name], farms[1][name]]) for name in farms[0]}
    town = np.zeros(C.N_SHOPS, dtype=np.int32)
    for shop in obs["town"]["unlocked_shops"]:
        town[C.SHOPS.index(shop)] += 1
    return State(
        **stacked,
        market_inventory=_dict_vector(obs["market"]["inventory"], C.PRODUCTS),
        town_shop_counts=town,
        step=np.asarray(obs.get("step", obs["day"] * turns_per_day + obs["hour"]), dtype=np.int32),
        rng_key=np.zeros(2, dtype=np.uint32),
    )


def _trace_choices(obs: dict, action: dict, rules: CacheRules) -> tuple[np.ndarray, np.ndarray]:
    """正規化済み行動をJAX固定候補表のindex列へ変換する。"""
    from kaggriculture.policy.torch import distribution as distribution

    env = distribution._envs_from_observations(
        [obs],
        rules.turns_per_day,
        rules.shed_capacity,
        rules.hire_mult,
        rules.max_market_orders,
    )[0]
    _, _, _, slots = distribution._teacher_force_trace(env, action)
    choices = np.zeros(L.MAX_DECODE_LEN, dtype=np.int32)
    mask = np.zeros(L.MAX_DECODE_LEN, dtype=np.bool_)
    for step, (candidates, slot_kind, selected) in enumerate(slots):
        candidate = candidates[selected]
        if slot_kind == "quantity":
            choice = selected
        elif slot_kind == "unit_op":
            op_name = distribution._op_name(candidate, V.ACTION_FARMER_OP, C.FARMER_OP_NAMES)
            item_name = distribution._item_name(candidate)
            op = C.FARMER_OP_NAMES.index(op_name)
            if op == C.FARMER_OP_PLANT:
                arg = C.CROPS.index(item_name)
            elif op in (C.FARMER_OP_PICKUP, C.FARMER_OP_PLACE):
                arg = C.SHED_ITEMS.index(item_name)
            else:
                arg = -1
            choice = _UNIT_CHOICE[(op, arg)]
        elif V.ACTION_MARKET_WAIT.start in candidate.index:
            choice = _MARKET_CHOICE[(C.N_MARKET_OPS, -1)]
        elif V.ACTION_MARKET_STOP.start in candidate.index:
            choice = _MARKET_CHOICE[(C.N_MARKET_OPS + 1, -1)]
        else:
            op_name = distribution._op_name(candidate, V.ACTION_MARKET_OP, C.MARKET_OP_NAMES)
            item_name = distribution._item_name(candidate)
            op = C.MARKET_OP_NAMES.index(op_name)
            if op == C.MARKET_OP_BUY_SEED:
                arg = C.CROPS.index(item_name)
            elif op == C.MARKET_OP_BUY_ANIMAL:
                arg = C.ANIMALS.index(item_name)
            elif op in (C.MARKET_OP_BUY_PRODUCT, C.MARKET_OP_SELL):
                arg = C.PRODUCTS.index(item_name)
            else:
                arg = -1
            choice = _MARKET_CHOICE[(op, arg)]
        choices[step] = choice
        mask[step] = True
    return choices, mask


def _cache_path(source: Path, cache_dir: Path, rules: CacheRules) -> Path:
    stat = source.stat()
    identity = (
        str(source.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        rules,
        _CACHE_VERSION,
    )
    digest = hashlib.sha256(repr(identity).encode()).hexdigest()
    return cache_dir / f"{digest}.npz"


def prepare_episode(source: Path, cache_dir: Path, rules: CacheRules) -> Path | None:
    """1 episodeを変換する。既存の同一cacheは再利用する。"""
    destination = _cache_path(source, cache_dir, rules)
    if destination.exists():
        return destination

    from kaggriculture.policy.torch import distribution as distribution
    from kaggriculture.training.bc.dataset import iter_replay_samples

    states = []
    players = []
    choices = []
    masks = []
    kwargs = {
        "turns_per_day": rules.turns_per_day,
        "shed_capacity": rules.shed_capacity,
        "hire_mult": rules.hire_mult,
        "max_market_orders": rules.max_market_orders,
    }
    for obs, action in iter_replay_samples(source, rules.min_player_reward):
        try:
            normalized = distribution.normalize_expert_action(obs, action, **kwargs)
            choice, mask = _trace_choices(obs, normalized, rules)
            states.append(observation_to_state(obs, turns_per_day=rules.turns_per_day))
            players.append(obs["player"])
            choices.append(choice)
            masks.append(mask)
        except (KeyError, TypeError, ValueError):
            continue
    if not states:
        return None

    arrays = {
        f"{_STATE_PREFIX}{name}": np.stack([np.asarray(getattr(state, name)) for state in states])
        for name in State._fields
    }
    arrays.update(
        player=np.asarray(players, dtype=np.int8),
        choices=np.stack(choices).astype(np.int16),
        decision_mask=np.stack(masks),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".tmp-{os.getpid()}")
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def prepare_episodes(
    sources: Sequence[Path], cache_dir: Path, rules: CacheRules
) -> tuple[list[Path], int]:
    """episode群を変換し、cache pathと新規作成数を返す。"""
    paths = []
    created = 0
    for source in sources:
        expected = _cache_path(source, cache_dir, rules)
        existed = expected.exists()
        path = prepare_episode(source, cache_dir, rules)
        if path is not None:
            paths.append(path)
            created += not existed
    return paths, created


def load_shard(path: Path) -> BCBatch:
    """1 cache shardをRAMへ読み込む。"""
    with np.load(path, allow_pickle=False) as data:
        states = State(*(data[f"{_STATE_PREFIX}{name}"] for name in State._fields))
        return BCBatch(
            states,
            data["player"].astype(np.int32),
            data["choices"].astype(np.int32),
            data["decision_mask"],
        )


def _take(batch: BCBatch, indices) -> BCBatch:
    states = State(*(field[indices] for field in batch.states))
    return BCBatch(
        states, batch.players[indices], batch.choices[indices], batch.decision_mask[indices]
    )


def _concat(left: BCBatch | None, right: BCBatch) -> BCBatch:
    if left is None:
        return right
    states = State(
        *(np.concatenate([a, b]) for a, b in zip(left.states, right.states, strict=True))
    )
    return BCBatch(
        states,
        np.concatenate([left.players, right.players]),
        np.concatenate([left.choices, right.choices]),
        np.concatenate([left.decision_mask, right.decision_mask]),
    )


def _prefetched_shards(paths: Sequence[Path]) -> Iterator[BCBatch]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = None
        for path in paths:
            next_future = executor.submit(load_shard, path)
            if future is not None:
                yield future.result()
            future = next_future
        if future is not None:
            yield future.result()


def iter_batches(
    paths: Sequence[Path], batch_size: int, *, seed: int, shuffle: bool, drop_last: bool
) -> Iterator[BCBatch]:
    """episode shardを先読みし、固定batchへまとめる。"""
    rng = np.random.default_rng(seed)
    ordered = list(paths)
    if shuffle:
        rng.shuffle(ordered)
    pending = None
    for shard in _prefetched_shards(ordered):
        if shuffle:
            shard = _take(shard, rng.permutation(len(shard.players)))
        pending = _concat(pending, shard)
        while len(pending.players) >= batch_size:
            yield _take(pending, slice(0, batch_size))
            pending = _take(pending, slice(batch_size, None))
    if pending is not None and len(pending.players) and not drop_last:
        yield pending
