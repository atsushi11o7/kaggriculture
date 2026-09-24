"""Opponent-pool persistence, selection, and evaluation reporting."""

import logging
import shutil
from pathlib import Path

import jax

from kaggriculture.training.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


def list_pool_members(pool_dir: Path) -> list[Path]:
    """Return pool member checkpoints in promotion order."""
    if not pool_dir.exists():
        return []
    members = [
        path
        for path in pool_dir.iterdir()
        if path.is_dir() and path.name.removeprefix("member_").isdigit()
    ]
    return sorted(members, key=lambda path: int(path.name.removeprefix("member_")))


def add_to_pool(pool_dir: Path, train_state, metadata: dict, pool_size: int) -> None:
    """Add a promoted candidate and prune the oldest excess members."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    existing = list_pool_members(pool_dir)
    next_index = 0
    if existing:
        next_index = max(int(path.name.removeprefix("member_")) for path in existing) + 1
    save_checkpoint(pool_dir / f"member_{next_index}", train_state, metadata)
    existing = list_pool_members(pool_dir)
    if pool_size > 0:
        for stale in existing[: max(0, len(existing) - pool_size)]:
            shutil.rmtree(stale)


def select_opponent(
    draw: float, has_pool: bool, anchor_probability: float, pool_probability: float
) -> str:
    """Select anchor, pool, or self from one uniform draw.

    Pool probability is assigned to the stable anchor while the pool is empty.
    """
    if not 0 <= anchor_probability <= 1 or not 0 <= pool_probability <= 1:
        raise ValueError("opponent probabilities must be between 0 and 1")
    if anchor_probability + pool_probability > 1:
        raise ValueError("anchor and pool probabilities must sum to at most 1")
    effective_anchor = anchor_probability if has_pool else anchor_probability + pool_probability
    if draw < effective_anchor:
        return "anchor"
    if has_pool and draw < anchor_probability + pool_probability:
        return "pool"
    return "self"


def evaluation_summary(result) -> dict[str, float | int]:
    """Summarize candidate-relative outcomes, cash, and PASS rates."""
    outcome = jax.device_get(result.outcome)
    cash = jax.device_get(result.cash)
    games_per_seat = outcome.shape[0] // 2

    def rate(values):
        return float(((values > 0) + 0.5 * (values == 0)).mean())

    return {
        "win_rate": float(jax.device_get(result.win_rate)),
        "wins": int((outcome > 0).sum()),
        "draws": int((outcome == 0).sum()),
        "losses": int((outcome < 0).sum()),
        "seat0_win_rate": rate(outcome[:games_per_seat]),
        "seat1_win_rate": rate(outcome[games_per_seat:]),
        "candidate_cash": float(cash[:, 0].mean()),
        "opponent_cash": float(cash[:, 1].mean()),
        "candidate_pass_rate": float(jax.device_get(result.pass_rate).mean()),
        "opponent_pass_rate": float(jax.device_get(result.opponent_pass_rate).mean()),
    }


def log_evaluation(update: int, opponent: str, result) -> dict[str, float | int]:
    """Log and return one evaluation summary."""
    summary = evaluation_summary(result)
    logger.info(
        "update=%d eval_opponent=%s win_rate=%.3f wins=%d draws=%d losses=%d "
        "seat0_win_rate=%.3f seat1_win_rate=%.3f candidate_cash=%.1f opponent_cash=%.1f "
        "candidate_pass_rate=%.3f opponent_pass_rate=%.3f",
        update,
        opponent,
        summary["win_rate"],
        summary["wins"],
        summary["draws"],
        summary["losses"],
        summary["seat0_win_rate"],
        summary["seat1_win_rate"],
        summary["candidate_cash"],
        summary["opponent_cash"],
        summary["candidate_pass_rate"],
        summary["opponent_pass_rate"],
    )
    return summary


def prune_step_checkpoints(directory: Path, keep_last: int) -> None:
    """Prune old step checkpoints while retaining best and pool directories."""
    if keep_last <= 0 or not directory.exists():
        return
    paths = sorted(
        (path for path in directory.glob("step_*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("step_")),
    )
    for stale in paths[: max(0, len(paths) - keep_last)]:
        shutil.rmtree(stale)


def opponent_for_choice(choice: str, anchor: dict, learner: dict, pool: dict | None = None) -> dict:
    """Return variables for an opponent class selected by :func:`select_opponent`."""
    if choice == "anchor":
        return anchor
    if choice == "self":
        return learner
    if choice == "pool" and pool is not None:
        return pool
    if choice == "pool":
        raise ValueError("pool opponent selected without pool variables")
    raise ValueError(f"unknown opponent choice: {choice}")
