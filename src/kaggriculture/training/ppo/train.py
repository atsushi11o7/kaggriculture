"""固定slot方策専用のGPU完結PPO学習入口。"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import hydra
import jax
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
    source_state = bc_core.create_train_state(source_model, source_variables, bc_core.BCConfig())
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
    variables = P.initialize(model, init_key)
    if cfg.ppo.init_bc_checkpoint and cfg.ppo.resume_checkpoint:
        raise ValueError("init_bc_checkpoint and resume_checkpoint are mutually exclusive")
    if cfg.ppo.init_bc_checkpoint:
        variables = _load_bc_actor(Path(to_absolute_path(cfg.ppo.init_bc_checkpoint)), variables)
    ppo_config = core.PPOConfig(
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_epsilon=cfg.ppo.clip_epsilon,
        value_clip_epsilon=cfg.ppo.value_clip_epsilon,
        value_coef=cfg.ppo.value_coef,
        entropy_coef=cfg.ppo.entropy_coef,
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
        metadata = read_checkpoint_metadata(directory)
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
        key, rollout_key, update_key, pool_key, member_key, seat_key, eval_key = jax.random.split(
            key, 7
        )
        started = time.perf_counter()
        pool_members = _pool_members(pool_dir)
        used_pool = bool(pool_members) and bool(
            jax.random.bernoulli(pool_key, cfg.ppo.pool_sample_prob)
        )
        if used_pool:
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
        else:
            rollout = collect_rollout(
                model,
                {"params": train_state.params},
                rollout_config,
                state,
                counters,
                rollout_key,
            )
        batch = to_ppo_batch(rollout, ppo_config.gamma, ppo_config.gae_lambda)
        train_state, metrics = core.update_epochs(
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
            logger.info(
                "update=%d seconds=%.1f samples/s=%.1f loss=%.4f invalid=%.3f clamped=%.3f pool=%s",
                update,
                elapsed,
                samples / elapsed,
                values.loss,
                invalid,
                clamped,
                used_pool,
            )

        metadata = {
            **checkpoint_shape_metadata(),
            "trainer": "ppo",
            "update": update,
            "model_config": asdict(model_config),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if cfg.ppo.eval_interval > 0 and update % cfg.ppo.eval_interval == 0:
            if not (best_dir / "metadata.json").exists():
                save_checkpoint(best_dir, train_state, metadata)
                _add_to_pool(pool_dir, train_state, metadata, cfg.ppo.pool_size)
                logger.info("update=%d promoted=bootstrap", update)
            else:
                best_variables = _opponent_variables(best_dir, train_state)
                result = evaluate_both_seats(
                    model,
                    {"params": train_state.params},
                    best_variables,
                    rollout_config,
                    eval_key,
                    cfg.ppo.eval_episodes,
                )
                win_rate = float(jax.device_get(result.win_rate))
                promoted = win_rate >= cfg.ppo.promotion_win_rate
                if promoted:
                    save_checkpoint(best_dir, train_state, metadata)
                    _add_to_pool(pool_dir, train_state, metadata, cfg.ppo.pool_size)
                logger.info("update=%d eval_win_rate=%.3f promoted=%s", update, win_rate, promoted)

        if update % cfg.ppo.checkpoint_interval == 0 or update == cfg.ppo.total_updates:
            directory = checkpoints_dir / f"step_{update}"
            save_checkpoint(directory, train_state, metadata)
            save_pytree(
                directory / "runtime.msgpack", {"key": key, "state": state, "counters": counters}
            )
            _prune_step_checkpoints(checkpoints_dir, cfg.ppo.keep_last_checkpoints)


if __name__ == "__main__":
    main()
