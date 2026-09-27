"""Shared observation days 3-5 tape, used before the shop key is known.

Extracted from a real day11-full-land DECEM replay (raw observation steps
72-143). Real top-player behavior in this window is essentially
shop-independent (verified across all 64 tapes in tape_bank.py: farmer and
hand commands match almost exactly regardless of eventual shop draw), so a
single representative tape covers this window for every shop key.
"""

from __future__ import annotations

import json
from pathlib import Path

_DATA_PATH = Path(__file__).with_name("bridge_tape_data.json")
BRIDGE_TAPE: tuple[dict, ...] = tuple(json.loads(_DATA_PATH.read_text()))
