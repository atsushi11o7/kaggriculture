"""Convert Kaggle observations into fixed-shape simulator states."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kaggriculture.policy.common import vocab as V
from kaggriculture.rules import constants as C
from kaggriculture.simulator.state import State


@dataclass(frozen=True)
class CacheRules:
    """expert正規化と候補列挙へ渡すゲーム規則。"""

    turns_per_day: int = 24
    shed_capacity: int = 100
    hire_mult: float = 1.0
    max_market_orders: int = C.MAX_MARKET_ORDERS
    min_player_reward: float | None = None


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


def paired_observations_to_state(
    observations: Sequence[dict], board_size: int = V.BOARD_SIZE, turns_per_day: int = 24
) -> State:
    """同一時点の両プレイヤー観測から完全なcritic用Stateを作る。

    Args:
        observations: プレイヤー0、1の順の観測。
        board_size: 盤面の一辺の長さ。
        turns_per_day: 1日あたりのターン数。

    Returns:
        両者のprivate情報を含むState。

    Raises:
        ValueError: 観測のプレイヤーまたは時点が一致しない場合。
    """
    if len(observations) != 2 or [int(obs["player"]) for obs in observations] != [0, 1]:
        raise ValueError("expected player 0 and player 1 observations")
    steps = [obs.get("step", obs["day"] * turns_per_day + obs["hour"]) for obs in observations]
    if steps[0] != steps[1]:
        raise ValueError("paired observations must describe the same step")
    obs = observations[0]
    farms = [
        _parse_farm(obs["farms"][player], observations[player]["private"], board_size)
        for player in range(2)
    ]
    stacked = {name: np.stack([farm[name] for farm in farms]) for name in farms[0]}
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
