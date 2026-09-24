"""Day<=11 opening controller.

Delegates the entire turn to the vendored hybrid2965 agent, except for land
purchases, which follow our own DSM-informed schedule (see land_rule.py).

hybrid2965 was chosen over cha22 (which wins head-to-head 12/12 against it)
because its natural day<=11 state -- animals (COW/GOOSE/SHEEP) bought and
placed early, multiple crops -- already matches the shared opening pattern
we found across real top-ladder replays (DECEM, Vadim Vasilenko, Unknown
Mother-Goose all converge on the same early animal investment; only DSM
differs). cha22 delays all animal investment to day 20+, and forcing it in
earlier via injected BUY_ANIMAL orders caused an uncontrolled cascade
(17-23 animals placed by day 11 instead of a handful) -- its internal
per-turn bookkeeping doesn't tolerate orders it didn't itself plan.
hybrid2965 needs no such patching for animals/crops, only the same land
override cha22 needed (its own pace, SW ~day11/SE ~day18, trails DSM's
day9/day10-11).
"""

from __future__ import annotations

from kaggriculture.policy.opening import hybrid2965_agent
from kaggriculture.policy.opening.land_rule import next_forced_purchase
from kaggriculture.rules import constants as C

OPENING_LAST_DAY = 11


def is_opening_day(day: int) -> bool:
    """Return whether this day is handled by the opening controller."""
    return day <= OPENING_LAST_DAY


def opening_action(observation: dict, configuration: dict | None = None) -> dict:
    """Return one turn's action for day<=11: hybrid2965's turn, with a land override.

    Through day 8, this is byte-for-byte hybrid2965 (it already buys NE on
    day 6, hour 7 every time, matching our target, so there's nothing to
    touch). From day 9 on, hybrid2965's own (slower) BUY_LAND order is
    dropped and replaced by our own schedule: BUY_LAND always targets
    "whichever quadrant is next", so leaving its own order in place would
    let it fire again right after ours in the same turn and buy an extra
    quadrant ahead of schedule.
    """
    action = hybrid2965_agent.agent(observation, configuration)
    if observation["day"] < 9:
        return action
    market = [order for order in action.get("market", []) if order[:1] != ["BUY_LAND"]]
    forced = next_forced_purchase(observation)
    if forced is not None:
        market = [forced, *market]
    action = dict(action, market=market[: C.MAX_MARKET_ORDERS])
    return action
