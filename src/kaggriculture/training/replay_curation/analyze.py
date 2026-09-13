"""Kaggleリプレイから選別用のプレイヤー別統計を抽出する。"""

from __future__ import annotations

import json
from pathlib import Path

from kaggriculture.training.replays.selection import SelectionEntry

_CROP_PRODUCTS = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"}
_ANIMAL_PRODUCTS = {"EGG", "MILK", "WOOL"}


def _strategy(sales: dict[str, int], buy_product_units: int) -> str:
    crop = sum(sales.get(item, 0) for item in _CROP_PRODUCTS)
    animal = sum(sales.get(item, 0) for item in _ANIMAL_PRODUCTS)
    total = sum(sales.values())
    if buy_product_units >= max(20, total // 3):
        return "market_trading"
    if animal > crop and animal >= total * 0.5:
        return "animal"
    if sales.get("MELON", 0) >= max(1, crop * 0.5):
        return "crop_melon"
    if crop >= total * 0.75:
        return "crop_mixed"
    if sales.get("FERTILIZER", 0) >= max(1, total * 0.25):
        return "fertilizer"
    return "mixed"


def analyze_episode(path: Path, rating: dict | None = None) -> list[SelectionEntry]:
    """1試合を解析し、両プレイヤー分の統計を返す。"""
    with open(path, encoding="utf-8") as stream:
        episode = json.load(stream)
    rewards = episode.get("rewards")
    steps = episode.get("steps", [])
    if not isinstance(rewards, list) or len(rewards) != 2 or not steps:
        raise ValueError("replay must contain two rewards and steps")
    if any(not isinstance(reward, (int, float)) for reward in rewards):
        raise ValueError("replay rewards must be numeric")
    if any(len(frame) != 2 for frame in steps):
        raise ValueError("replay must have two players in every step")
    names = [None, None]
    agents = episode.get("info", {}).get("Agents", [])
    for player, agent in enumerate(agents[:2]):
        names[player] = agent.get("Name")
    stat = path.stat()
    sales = [{}, {}]
    buy_product_units = [0, 0]
    hires = [0, 0]
    land = [0, 0]
    max_hands = [0, 0]
    for frame in steps:
        for player, record in enumerate(frame):
            observation = record.get("observation") or {}
            farms = observation.get("farms") or []
            if len(farms) > player:
                max_hands[player] = max(max_hands[player], len(farms[player].get("hands", [])))
            action = record.get("action") or {}
            for order in action.get("market", []):
                if not isinstance(order, list) or not order:
                    continue
                op = order[0]
                if op == "SELL" and len(order) >= 3:
                    sales[player][order[1]] = sales[player].get(order[1], 0) + max(int(order[2]), 0)
                elif op == "BUY_PRODUCT" and len(order) >= 3:
                    buy_product_units[player] += max(int(order[2]), 0)
                elif op == "HIRE":
                    hires[player] += 1
                elif op == "BUY_LAND":
                    land[player] += 1
    result = []
    for player in range(2):
        own = float(rewards[player])
        other = float(rewards[1 - player])
        result.append(
            SelectionEntry(
                episode_id=str(episode.get("id") or path.stem),
                path=str(path.resolve()),
                player=player,
                split="",
                terminal_cash=own,
                opponent_cash=other,
                margin=own - other,
                avg_agent_score=float(rating["avg_score"]) if rating else None,
                min_agent_score=float(rating["min_score"]) if rating else None,
                create_time=rating.get("create_time") if rating else None,
                agent_name=names[player],
                opponent_name=names[1 - player],
                strategy=_strategy(sales[player], buy_product_units[player]),
                max_hands=max_hands[player],
                land_purchases=land[player],
                hires=hires[player],
                sell_units=sum(sales[player].values()),
                buy_product_units=buy_product_units[player],
                source_size=stat.st_size,
                source_mtime_ns=stat.st_mtime_ns,
            )
        )
    return result
