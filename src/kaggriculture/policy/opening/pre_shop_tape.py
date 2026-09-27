"""Frozen shop-independent action prefix for days 0 through 2.

Extracted from the same real DECEM day11-full-land replay used as
bridge_tape.py's source (see tape_bank.py's "BAKERY|BAKERY" entry), so the
day 2 -> day 3 handoff is internally consistent: the bridge tape's day 3-5
quantities were scripted assuming exactly this day 0-2 history, not some
unrelated game's. Splicing a different source's day 0-2 in here (as the
previous hybrid2965-derived tape did) reintroduces the same kind of
cash/state mismatch that made the day5 -> day6 tape lock-in unreliable.
"""

from __future__ import annotations

import json
from pathlib import Path

_DATA_PATH = Path(__file__).with_name("pre_shop_tape_data.json")
PRE_SHOP_TAPE: tuple[dict, ...] = tuple(json.loads(_DATA_PATH.read_text()))
