"""Land-purchase override for the day<=11 opening window.

hybrid2965's own land timing (NE day6, SW day11, SE day18) trails the real
top-ladder pace observed from DSM replays (NE day6, SW day9, SE day10-11,
consistently full-owned). Land purchase is a rare (<=3 per game), high-value
decision, so we buy on this faster schedule directly rather than trusting
either the network or hybrid2965's own pacing for it.

Cash is not the limiting factor at this pace (verified: DSM already holds
1900+ well before each purchase), but the day/cash guard is kept as a safety
net for unusual games. The land order is inserted ahead of hybrid2965's own
market orders (see controller.py), so it is charged before hybrid2965's own
turn spending sees the remaining cash; a fixed operating reserve is required
on top of the sticker price so a same-turn purchase can't crowd out its own
seed/hire/feed orders that turn.
"""

from __future__ import annotations

# (quadrant, day it becomes eligible, price) in the fixed purchase order.
# NE is left to hybrid2965's own market order: it already buys NE on day 6,
# hour 7 every time (verified, zero variance across seeds), matching our
# target, so there is nothing to override there. Only SW/SE need forcing.
_SCHEDULE = (("SW", 9, 2000), ("SE", 11, 4000))

# Headroom kept on top of the sticker price so the same-turn forced purchase
# doesn't starve hybrid2965's own seed/hire/feed orders for that turn.
_OPERATING_RESERVE = 300


def next_forced_purchase(obs: dict) -> list | None:
    """Return a BUY_LAND market entry if the schedule calls for one this turn.

    Returns None when the next quadrant in order isn't due yet, or its price
    plus operating reserve isn't affordable yet (in which case the purchase
    is deferred, not skipped: the same quadrant is retried on a later turn).
    """
    player = obs["player"]
    farm = obs["farms"][player]
    owned = set(farm["unlocked_quadrants"])
    day = obs["day"]
    cash = farm["money"]

    for quadrant, eligible_day, price in _SCHEDULE:
        if quadrant in owned:
            continue
        if day >= eligible_day and cash >= price + _OPERATING_RESERVE:
            return ["BUY_LAND"]
        return None
    return None
