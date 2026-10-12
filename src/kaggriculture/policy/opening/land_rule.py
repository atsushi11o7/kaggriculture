"""Kaggleの0始まりraw dayに基づく土地購入期限。"""

from __future__ import annotations

# 上位プレイヤーのリプレイ観測から直接求めた、最頻の土地購入日。
# 各期限以降も再試行し、所持金による1日の遅れを許容する。
LAND_PURCHASE_SCHEDULE = (("NE", 6, 1000), ("SW", 9, 2000), ("SE", 10, 4000))
OPERATING_RESERVE = 300


_RESERVE_LEAD_DAYS = 1


def required_capital(obs: dict, purchases_ahead: int = 0) -> int:
    """予定された土地購入のために確保する所持金を返す。

    購入可能になる前日からのみ資金を確保する。リプレイでは、上位プレイヤーは
    数日前から全額を貯めず、購入直前まで作物や動物へ投資していた
    （day 8の所持金がSW価格を下回ることも多い）。
    """
    player = obs["player"]
    owned = set(obs["farms"][player]["unlocked_quadrants"])
    pending = [entry for entry in LAND_PURCHASE_SCHEDULE if entry[0] not in owned]
    if purchases_ahead >= len(pending):
        return 0
    _, eligible_day, price = pending[purchases_ahead]
    if obs["day"] < eligible_day - _RESERVE_LEAD_DAYS:
        return 0
    return price + OPERATING_RESERVE


def next_forced_purchase(obs: dict, available_cash: float | None = None) -> list | None:
    """raw observation dayの期限を迎え、購入可能ならBUY_LANDを返す。"""
    player = obs["player"]
    farm = obs["farms"][player]
    owned = set(farm["unlocked_quadrants"])
    day = obs["day"]
    cash = farm["money"] if available_cash is None else available_cash

    for quadrant, eligible_day, price in LAND_PURCHASE_SCHEDULE:
        if quadrant in owned:
            continue
        if day >= eligible_day and cash >= price + OPERATING_RESERVE:
            return ["BUY_LAND"]
        return None
    return None
