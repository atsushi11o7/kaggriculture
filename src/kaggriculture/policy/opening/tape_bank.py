"""Shop-keyed observation days 6-11 tapes, one per (day3, day6) shop pair.

Each entry is extracted from a real day11-full-land DECEM replay whose day 3
and day 6 shop draws match the key, covering raw observation steps 144-287.
Weed-reactive DIGs were trimmed at extraction time (see the session's
extract_tape.py); the live weed_repair layer in tape_repair.py reinserts an
equivalent DIG dynamically wherever a *different* game's weeds land.
"""

from __future__ import annotations

import json
from pathlib import Path

_DATA_PATH = Path(__file__).with_name("tape_bank_data.json")
_RAW: dict[str, list[dict]] = json.loads(_DATA_PATH.read_text())

TAPE_BANK: dict[str, tuple[dict, ...]] = {key: tuple(tape) for key, tape in _RAW.items()}

# Every one of the 64 (day3, day6) shop-pair combinations has its own tape;
# this is only a defensive fallback in case shops.py's shop list ever grows.
DEFAULT_KEY = next(iter(TAPE_BANK))


def shop_key(day3_shop: str, day6_shop: str) -> str:
    return f"{day3_shop}|{day6_shop}"


def lookup(day3_shop: str, day6_shop: str) -> tuple[dict, ...]:
    """Return the closest-matching tape for a (day3, day6) shop pair."""
    key = shop_key(day3_shop, day6_shop)
    if key in TAPE_BANK:
        return TAPE_BANK[key]
    # Shops revealed in an unexpected order or an unseen combination: fall
    # back to a tape sharing at least the day6 shop, else any tape.
    for candidate_key, tape in TAPE_BANK.items():
        if candidate_key.endswith(f"|{day6_shop}"):
            return tape
    return TAPE_BANK[DEFAULT_KEY]
