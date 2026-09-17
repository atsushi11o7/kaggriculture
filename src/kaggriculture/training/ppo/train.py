"""固定slot方策専用のGPU完結PPO学習入口。"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import hydra
import jax
import optax
from flax import traverse_util
from flax.core import freeze, unfreeze
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from kaggriculture.policy.common.config import (
    ModelConfig,
    checkpoint_shape_metadata,
    validate_checkpoint_metadata,
)
from kaggriculture.policy.jax import history as H
from kaggriculture.policy.jax import model as M
from kaggriculture.policy.jax import policy as P
from kaggriculture.simulator.reset import reset
from kaggriculture.training.bc import core as bc_core
from kaggriculture.training.checkpoint import (
    load_checkpoint,
    load_pytree,
    read_checkpoint_metadata,
    save_checkpoint,
    save_pytree,
)
from kaggriculture.training.ppo import core
from kaggriculture.training.ppo.evaluation import evaluate_both_seats
from kaggriculture.training.ppo.rollout import (
    RolloutConfig,
    collect_rollout,
    collect_rollout_vs_opponent,
    to_ppo_batch,
)

logger = logging.getLogger(__name__)


def _load_bc_actor(path: Path, variables: dict) -> dict:
    """固定slot BCの同名・同shape Actor parameterだけを読み込む。"""
    metadata = read_checkpoint_metadata(path)
    validate_checkpoint_metadata(metadata)
    source_config = ModelConfig(**metadata["model_config"])
    source_model = M.PolicyValueNet(source_config)
    source_variables = P.initialize(source_model, jax.random.key(0))
    # bc/train.pyはlearning_rateを常にcosine decay schedule(callable)として
    # optimizerへ渡すため、復元用テンプレートもoptax内部のstate構造(count等)を
    # 合わせる必要がある(paramsだけ取り出してすぐ捨てるので値は使わない)。
    dummy_schedule = optax.cosine_decay_schedule(1.0, 1)
    source_state = bc_core.create_train_state(
        source_model, source_variables, bc_core.BCConfig(dummy_schedule)
    )
    source_state, _ = load_checkpoint(path, source_state)
    target = traverse_util.flatten_dict(unfreeze(variables["params"]))
    source = traverse_util.flatten_dict(unfreeze(source_state.params))
    for name, value in source.items():
        if name in target and target[name].shape == value.shape and name[0] != "value_head":
            target[name] = value
    return {"params": freeze(traverse_util.unflatten_dict(target))}


def _opponent_variables(directory: Path, train_state) -> dict:
    """train_stateと同じ構造をtemplateにして、checkpointのparamsだけを取り出す。"""
    metadata = read_checkpoint_metadata(directory)
    validate_checkpoint_metadata(metadata)
    restored, _ = load_checkpoint(directory, train_state)
    return {"params": restored.params}


def _pool_members(pool_dir: Path) -> list[Path]:
    if not pool_dir.exists():
        return []
    members = [
        path
        for path in pool_dir.iterdir()
        if path.is_dir() and path.name.removeprefix("member_").isdigit()
    ]
    return sorted(members, key=lambda path: int(path.name.removeprefix("member_")))


def _add_to_pool(pool_dir: Path, train_state, metadata: dict, pool_size: int) -> None:
    """昇格したcandidateをpoolへ追加し、古い順にpool_size件まで間引く。"""
    pool_dir.mkdir(parents=True, exist_ok=True)
    existing = _pool_members(pool_dir)
    next_index = 0
    if existing:
        next_index = max(int(path.name.removeprefix("member_")) for path in existing) + 1
    save_checkpoint(pool_dir / f"member_{next_index}", train_state, metadata)
    existing = _pool_members(pool_dir)
    if pool_size > 0:
        for stale in existing[: max(0, len(existing) - pool_size)]:
            shutil.rmtree(stale)


def _select_opponent(
    draw: float, has_pool: bool, anchor_probability: float, pool_probability: float
) -> str:
    """Select the rollout opponent class from one uniform draw.

    poolが空の間は、pool分の確率もself-playへ逃がさずanchorへ回す
    (両者が同時に劣化しうるself-playを増やさないため)。
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


def _evaluation_summary(result) -> dict[str, float | int]:
    """Summarize candidate-relative outcomes and cash for logging."""
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
    }


def _log_evaluation(update: int, opponent: str, result) -> dict[str, float | int]:
    summary = _evaluation_summary(result)
    logger.info(
        "update=%d eval_opponent=%s win_rate=%.3f wins=%d draws=%d losses=%d "
        "seat0_win_rate=%.3f seat1_win_rate=%.3f candidate_cash=%.1f opponent_cash=%.1f",
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
    )
    return summary


def _prune_step_checkpoints(directory: Path, keep_last: int) -> None:
    """checkpoints/step_*だけ古い順に間引く(best/poolは対象外)。"""
    if keep_last <= 0 or not directory.exists():
        return
    paths = sorted(
        (path for path in directory.glob("step_*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("step_")),
    )
    for stale in paths[: max(0, len(paths) - keep_last)]:
        shutil.rmtree(stale)


def _rollout_config(cfg):
    rules = cfg.rules
    return RolloutConfig(
        horizon=cfg.ppo.rollout_horizon,
        board_size=rules.board_size,
        turns_per_day=rules.turns_per_day,
        shed_capacity=rules.shed_capacity,
        weed_chance=rules.weed_chance,
        shop_unlock_interval=rules.shop_unlock_interval,
        shop_sell_interval=rules.shop_sell_interval,
        center_sell_interval=rules.center_sell_interval,
        hire_mult=rules.hire_mult,
        max_shop_instances=rules.max_shop_instances,
        episode_steps=rules.episode_steps,
        starting_money=rules.starting_money,
        temperature=cfg.ppo.temperature,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="ppo")
def main(cfg: DictConfig) -> None:
    samples = cfg.env.batch_size * cfg.ppo.rollout_horizon * 2
    if samples % cfg.ppo.minibatch_size:
        raise ValueError("minibatch_size must divide batch_size * horizon * 2")
    model_config = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    model = M.PolicyValueNet(model_config)
    key = jax.random.key(cfg.ppo.seed)
    key, init_key, reset_key = jax.random.split(key, 3)
    initial_variables = P.initialize(model, init_key)
    variables = initial_variables
    if cfg.ppo.init_bc_checkpoint and cfg.ppo.resume_checkpoint:
        raise ValueError("init_bc_checkpoint and resume_checkpoint are mutually exclusive")
    resume_metadata = None
    if cfg.ppo.resume_checkpoint:
        resume_metadata = read_checkpoint_metadata(
            Path(to_absolute_path(cfg.ppo.resume_checkpoint))
        )
    anchor_checkpoint = cfg.ppo.anchor_bc_checkpoint or cfg.ppo.init_bc_checkpoint
    if anchor_checkpoint is None and resume_metadata is not None:
        previous_ppo = resume_metadata.get("config", {}).get("ppo", {})
        anchor_checkpoint = previous_ppo.get("anchor_bc_checkpoint") or previous_ppo.get(
            "init_bc_checkpoint"
        )
    if anchor_checkpoint is None:
        raise ValueError("PPO requires init_bc_checkpoint or anchor_bc_checkpoint")
    anchor_path = Path(to_absolute_path(anchor_checkpoint))
    anchor_variables = _load_bc_actor(anchor_path, initial_variables)
    if cfg.ppo.init_bc_checkpoint:
        variables = _load_bc_actor(
            Path(to_absolute_path(cfg.ppo.init_bc_checkpoint)), initial_variables
        )
    ppo_config = core.PPOConfig(
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_epsilon=cfg.ppo.clip_epsilon,
        value_clip_epsilon=cfg.ppo.value_clip_epsilon,
        value_coef=cfg.ppo.value_coef,
        entropy_coef=cfg.ppo.entropy_coef,
        target_kl=cfg.ppo.target_kl,
        max_grad_norm=cfg.ppo.max_grad_norm,
        learning_rate=cfg.ppo.learning_rate,
        weight_decay=cfg.ppo.weight_decay,
        temperature=cfg.ppo.temperature,
        turns_per_day=cfg.rules.turns_per_day,
        shed_capacity=cfg.rules.shed_capacity,
    )
    train_state = core.create_train_state(model, variables, ppo_config)
    state = reset(reset_key, cfg.env.batch_size, starting_money=cfg.rules.starting_money)
    counters = H.zeros(cfg.env.batch_size)
    start_update = 0
    if cfg.ppo.resume_checkpoint:
        directory = Path(to_absolute_path(cfg.ppo.resume_checkpoint))
        metadata = resume_metadata
        validate_checkpoint_metadata(metadata)
        train_state, _ = load_checkpoint(directory, train_state)
        start_update = int(metadata["update"])
        runtime = load_pytree(
            directory / "runtime.msgpack", {"key": key, "state": state, "counters": counters}
        )
        key, state, counters = runtime["key"], runtime["state"], runtime["counters"]
    rollout_config = _rollout_config(cfg)
    checkpoints_dir = Path("checkpoints")
    best_dir = checkpoints_dir / "best"
    # best/step_*と同じくHydraの実行ディレクトリ配下の相対名として扱う
    # (to_absolute_pathでchdir前の元cwd基準にすると、実行のたびに共有される
    # 固定パスになってしまい、別runのpool対戦相手が混ざり込む)。
    pool_dir = checkpoints_dir / cfg.ppo.pool_dir
    for update in range(start_update + 1, cfg.ppo.total_updates + 1):
        key, rollout_key, update_key, opponent_key, member_key, seat_key, eval_key = (
            jax.random.split(key, 7)
        )
        started = time.perf_counter()
        pool_members = _pool_members(pool_dir)
        opponent = _select_opponent(
            float(jax.random.uniform(opponent_key)),
            bool(pool_members),
            cfg.ppo.anchor_sample_prob,
            cfg.ppo.pool_sample_prob,
        )
        if opponent == "self":
            rollout = collect_rollout(
                model,
                {"params": train_state.params},
                rollout_config,
                state,
                counters,
                rollout_key,
            )
        else:
            if opponent == "anchor":
                opponent_variables = anchor_variables
            else:
                member = pool_members[int(jax.random.randint(member_key, (), 0, len(pool_members)))]
                opponent_variables = _opponent_variables(member, train_state)
            learner_seat = int(jax.random.bernoulli(seat_key))
            rollout = collect_rollout_vs_opponent(
                model,
                {"params": train_state.params},
                opponent_variables,
                learner_seat,
                rollout_config,
                state,
                counters,
                rollout_key,
            )
        batch = to_ppo_batch(rollout, ppo_config.gamma, ppo_config.gae_lambda)
        train_state, metrics, epochs_completed = core.update_epochs(
            model,
            train_state,
            batch,
            update_key,
            ppo_config,
            cfg.ppo.update_epochs,
            cfg.ppo.minibatch_size,
        )
        jax.block_until_ready(metrics.loss)
        state, counters = rollout.final_state, rollout.final_counters
        if update % cfg.ppo.log_interval == 0:
            elapsed = time.perf_counter() - started
            values = jax.device_get(metrics)
            invalid = float(jax.device_get(rollout.executor_invalid).mean())
            clamped = float(jax.device_get(rollout.executor_clamped).mean())
            invalid_unit = float(jax.device_get(rollout.executor_invalid_unit).mean())
            clamped_unit = float(jax.device_get(rollout.executor_clamped_unit).mean())
            dones = jax.device_get(rollout.dones)
            terminal_count = int(dones.sum())
            value_arr = jax.device_get(batch.old_value)
            return_arr = jax.device_get(batch.returns)
            advantage_arr = jax.device_get(batch.advantages)
            logger.info(
                "update=%d seconds=%.1f samples/s=%.1f loss=%.4f policy_loss=%.4f "
                "value_loss=%.4f entropy=%.4f approx_kl=%.4f clip_fraction=%.3f epochs=%d "
                "terminal_count=%d "
                "value_mean=%.4f value_std=%.4f return_mean=%.4f return_std=%.4f "
                "advantage_mean=%.4f advantage_std=%.4f "
                "invalid=%.3f clamped=%.3f invalid_unit=%.3f clamped_unit=%.3f opponent=%s",
                update,
                elapsed,
                samples / elapsed,
                values.loss,
                values.policy_loss,
                values.value_loss,
                values.entropy,
                values.approx_kl,
                values.clip_fraction,
                int(jax.device_get(epochs_completed)),
                terminal_count,
                value_arr.mean(),
                value_arr.std(),
                return_arr.mean(),
                return_arr.std(),
                advantage_arr.mean(),
                advantage_arr.std(),
                invalid,
                clamped,
                invalid_unit,
                clamped_unit,
                opponent,
            )

        metadata = {
            **checkpoint_shape_metadata(),
            "trainer": "ppo",
            "update": update,
            "model_config": asdict(model_config),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if cfg.ppo.eval_interval > 0 and update % cfg.ppo.eval_interval == 0:
            best_exists = (best_dir / "metadata.json").exists()
            best_variables = (
                _opponent_variables(best_dir, train_state) if best_exists else anchor_variables
            )
            best_key, anchor_key = jax.random.split(eval_key)
            best_result = evaluate_both_seats(
                model,
                {"params": train_state.params},
                best_variables,
                rollout_config,
                best_key,
                cfg.ppo.eval_episodes,
            )
            best_summary = _log_evaluation(
                update, "best" if best_exists else "anchor_as_best", best_result
            )
            if best_exists:
                anchor_result = evaluate_both_seats(
                    model,
                    {"params": train_state.params},
                    anchor_variables,
                    rollout_config,
                    anchor_key,
                    cfg.ppo.eval_episodes,
                )
                anchor_summary = _log_evaluation(update, "anchor", anchor_result)
            else:
                anchor_summary = best_summary
            promoted = (
                best_summary["win_rate"] >= cfg.ppo.promotion_win_rate
                and anchor_summary["win_rate"] >= cfg.ppo.anchor_promotion_win_rate
            )
            if promoted:
                save_checkpoint(best_dir, train_state, metadata)
                _add_to_pool(pool_dir, train_state, metadata, cfg.ppo.pool_size)
            logger.info(
                "update=%d promoted=%s best_win_rate=%.3f anchor_win_rate=%.3f",
                update,
                promoted,
                best_summary["win_rate"],
                anchor_summary["win_rate"],
            )

        if update % cfg.ppo.checkpoint_interval == 0 or update == cfg.ppo.total_updates:
            directory = checkpoints_dir / f"step_{update}"
            save_checkpoint(directory, train_state, metadata)
            save_pytree(
                directory / "runtime.msgpack", {"key": key, "state": state, "counters": counters}
            )
            _prune_step_checkpoints(checkpoints_dir, cfg.ppo.keep_last_checkpoints)


if __name__ == "__main__":
    main()
