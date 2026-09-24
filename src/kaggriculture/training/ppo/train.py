"""固定slot方策専用のGPU完結PPO学習入口。"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, replace
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from kaggriculture.policy.common.config import (
    ModelConfig,
    checkpoint_shape_metadata,
    validate_checkpoint_metadata,
)
from kaggriculture.policy.jax import history as H
from kaggriculture.policy.jax import policy as P
from kaggriculture.policy.jax.model_factory import (
    create_model,
    critic_version,
)
from kaggriculture.simulator.reset import reset
from kaggriculture.training.checkpoint import (
    load_checkpoint,
    load_pytree,
    read_checkpoint_metadata,
    save_checkpoint,
    save_pytree,
)
from kaggriculture.training.ppo import core, full_game
from kaggriculture.training.ppo.checkpointing import (
    load_actor_checkpoint,
    load_initial_bc_checkpoint,
    load_opponent_variables,
    load_value_checkpoint,
    validate_critic_architecture,
)
from kaggriculture.training.ppo.evaluation import evaluate_both_seats
from kaggriculture.training.ppo.population import (
    add_to_pool,
    list_pool_members,
    log_evaluation,
    opponent_for_choice,
    prune_step_checkpoints,
    select_opponent,
)
from kaggriculture.training.ppo.rollout import (
    RolloutConfig,
    collect_rollout,
    collect_rollout_vs_opponent,
    to_critic_batch,
    to_ppo_batch,
)

logger = logging.getLogger(__name__)


def _validate_training_config(cfg) -> None:
    """Validate relationships between PPO batch and warm-up settings."""
    if cfg.ppo.critic_warmup_updates < 0:
        raise ValueError("critic_warmup_updates must be nonnegative")
    if cfg.ppo.critic_warmup_updates > 0:
        if cfg.ppo.critic_warmup_epochs <= 0:
            raise ValueError("critic_warmup_epochs must be positive")
        if cfg.ppo.rollout_horizon < cfg.rules.episode_steps:
            raise ValueError("critic warm-up requires rollout_horizon >= episode_steps")
        critic_samples = cfg.env.batch_size * cfg.ppo.rollout_horizon
        if critic_samples % cfg.ppo.minibatch_size:
            raise ValueError(
                "minibatch_size must divide batch_size * horizon during critic warm-up"
            )
    samples = cfg.env.batch_size * cfg.ppo.rollout_horizon * 2
    if samples % cfg.ppo.minibatch_size:
        raise ValueError("minibatch_size must divide batch_size * horizon * 2")


def _checkpoint_metadata(cfg, model_config, reference_path: Path, **progress) -> dict:
    """Build metadata shared by warm-up and PPO checkpoints."""
    return {
        **checkpoint_shape_metadata(),
        "trainer": "ppo",
        "critic_architecture_version": critic_version(),
        "model_config": asdict(model_config),
        "reference_checkpoint_resolved": str(reference_path.resolve()),
        "config": OmegaConf.to_container(cfg, resolve=True),
        **progress,
    }


def _save_runtime(directory: Path, key, state, counters, daily_margin) -> None:
    """Save the environment state needed for an exact training resume."""
    save_pytree(
        directory / "runtime.msgpack",
        {"key": key, "state": state, "counters": counters, "daily_margin": daily_margin},
    )


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
        daily_reward_coefficient=cfg.ppo.daily_reward_coefficient,
        daily_reward_scale=cfg.ppo.daily_reward_scale,
        daily_reward_maximum=cfg.ppo.daily_reward_maximum,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="ppo")
def main(cfg: DictConfig) -> None:
    _validate_training_config(cfg)
    samples = cfg.env.batch_size * cfg.ppo.rollout_horizon * 2
    model_config = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    model = create_model(model_config)
    key = jax.random.key(cfg.ppo.seed)
    key, init_key, reset_key = jax.random.split(key, 3)
    initial_variables = P.initialize(model, init_key)
    variables = initial_variables
    if (
        sum(
            bool(path)
            for path in (
                cfg.ppo.init_bc_checkpoint,
                cfg.ppo.init_value_checkpoint,
                cfg.ppo.resume_checkpoint,
            )
        )
        > 1
    ):
        raise ValueError(
            "init_bc_checkpoint, init_value_checkpoint and resume_checkpoint are mutually exclusive"
        )
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
    anchor_variables = load_actor_checkpoint(anchor_path, initial_variables, model_config)
    if cfg.ppo.init_bc_checkpoint:
        variables = load_initial_bc_checkpoint(
            Path(to_absolute_path(cfg.ppo.init_bc_checkpoint)),
            initial_variables,
            model_config,
            cfg.ppo.gamma,
            cfg.ppo.daily_reward_coefficient,
            cfg.ppo.daily_reward_scale,
            cfg.ppo.daily_reward_maximum,
            allow_reward_mismatch=(
                cfg.ppo.critic_warmup_updates > 0 or cfg.ppo.allow_value_reward_mismatch
            ),
        )
    if cfg.ppo.init_value_checkpoint:
        variables = load_value_checkpoint(
            Path(to_absolute_path(cfg.ppo.init_value_checkpoint)),
            initial_variables,
            model_config,
            cfg.ppo.gamma,
            cfg.ppo.daily_reward_coefficient,
            cfg.ppo.daily_reward_scale,
            cfg.ppo.daily_reward_maximum,
            allow_reward_mismatch=(
                cfg.ppo.critic_warmup_updates > 0 or cfg.ppo.allow_value_reward_mismatch
            ),
        )
    reference_checkpoint = cfg.ppo.reference_checkpoint
    if reference_checkpoint is None and resume_metadata is not None:
        reference_checkpoint = resume_metadata.get(
            "reference_checkpoint_resolved"
        ) or resume_metadata.get("config", {}).get("ppo", {}).get("reference_checkpoint")
    reference_path = Path(to_absolute_path(reference_checkpoint or anchor_checkpoint))
    reference_variables = (
        anchor_variables
        if reference_path == anchor_path
        else load_actor_checkpoint(reference_path, initial_variables, model_config)
    )
    ppo_config = core.PPOConfig(
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_epsilon=cfg.ppo.clip_epsilon,
        value_clip_epsilon=cfg.ppo.value_clip_epsilon,
        value_coef=cfg.ppo.value_coef,
        entropy_coef=cfg.ppo.entropy_coef,
        reference_actor_l2_coef=cfg.ppo.reference_actor_l2_coef,
        target_kl=cfg.ppo.target_kl,
        max_grad_norm=cfg.ppo.max_grad_norm,
        learning_rate=cfg.ppo.learning_rate,
        weight_decay=cfg.ppo.weight_decay,
        temperature=cfg.ppo.temperature,
        turns_per_day=cfg.rules.turns_per_day,
        shed_capacity=cfg.rules.shed_capacity,
    )
    state = reset(reset_key, cfg.env.batch_size, starting_money=cfg.rules.starting_money)
    counters = H.zeros(cfg.env.batch_size)
    daily_margin = jnp.zeros((cfg.env.batch_size,), dtype=jnp.float32)
    start_update = 0
    warmup_start = 0
    train_state = None
    warmup_state = None
    resume_phase = resume_metadata.get("phase", "ppo") if resume_metadata else None
    if cfg.ppo.resume_checkpoint:
        directory = Path(to_absolute_path(cfg.ppo.resume_checkpoint))
        metadata = resume_metadata
        validate_checkpoint_metadata(metadata)
        validate_critic_architecture(metadata, directory)
        if resume_phase == "critic_warmup":
            warmup_state = core.create_critic_train_state(
                model,
                initial_variables,
                learning_rate=cfg.ppo.critic_warmup_learning_rate,
                max_grad_norm=cfg.ppo.max_grad_norm,
            )
            warmup_state, _ = load_checkpoint(directory, warmup_state)
            warmup_start = int(metadata["warmup_update"])
        else:
            train_state = core.create_train_state(model, variables, ppo_config)
            train_state, _ = load_checkpoint(directory, train_state)
            start_update = int(metadata["update"])
        runtime_target = {
            "key": key,
            "state": state,
            "counters": counters,
            "daily_margin": daily_margin,
        }
        runtime = load_pytree(directory / "runtime.msgpack", runtime_target)
        key, state, counters, daily_margin = (
            runtime["key"],
            runtime["state"],
            runtime["counters"],
            runtime["daily_margin"],
        )

    rollout_config = _rollout_config(cfg)
    checkpoints_dir = Path("checkpoints")
    best_dir = checkpoints_dir / "best"
    run_warmup = resume_phase == "critic_warmup" or (
        resume_metadata is None and cfg.ppo.critic_warmup_updates > 0
    )
    if run_warmup:
        if warmup_state is None:
            warmup_state = core.create_critic_train_state(
                model,
                variables,
                learning_rate=cfg.ppo.critic_warmup_learning_rate,
                max_grad_norm=cfg.ppo.max_grad_norm,
            )
        rollout = critic_batch = metrics = before = None
        for warmup_update in range(warmup_start + 1, cfg.ppo.critic_warmup_updates + 1):
            key, rollout_key, update_key, seat_key = jax.random.split(key, 4)
            started = time.perf_counter()
            learner_seat = int(jax.random.bernoulli(seat_key))
            rollout = collect_rollout_vs_opponent(
                model,
                {"params": warmup_state.params},
                anchor_variables,
                learner_seat,
                rollout_config,
                state,
                counters,
                rollout_key,
                daily_margin,
            )
            critic_batch = to_critic_batch(rollout, ppo_config.gamma)
            valid_values = int(jax.device_get(critic_batch.mask.sum()))
            if valid_values == 0:
                raise ValueError("critic warm-up rollout contains no completed episode")
            before = core.evaluate_critic(
                model,
                warmup_state.params,
                critic_batch,
                cfg.rules.turns_per_day,
                cfg.ppo.minibatch_size,
            )
            warmup_state, _ = core.update_critic_epochs(
                model,
                warmup_state,
                critic_batch,
                update_key,
                cfg.ppo.critic_warmup_epochs,
                cfg.ppo.minibatch_size,
                cfg.rules.turns_per_day,
            )
            metrics = core.evaluate_critic(
                model,
                warmup_state.params,
                critic_batch,
                cfg.rules.turns_per_day,
                cfg.ppo.minibatch_size,
            )
            jax.block_until_ready(metrics.loss)
            state, counters, daily_margin = (
                rollout.final_state,
                rollout.final_counters,
                rollout.final_margin,
            )
            elapsed = time.perf_counter() - started
            values, values_before = jax.device_get((metrics, before))
            logger.info(
                "phase=critic_warmup update=%d/%d seconds=%.1f samples/s=%.1f "
                "valid_values=%d loss=%.4f r2=%.4f correlation=%.4f "
                "r2_before_update=%.4f correlation_before_update=%.4f",
                warmup_update,
                cfg.ppo.critic_warmup_updates,
                elapsed,
                valid_values / elapsed,
                valid_values,
                values.loss,
                values.r2,
                values.correlation,
                values_before.r2,
                values_before.correlation,
            )
            metadata = _checkpoint_metadata(
                cfg,
                model_config,
                reference_path,
                phase="critic_warmup",
                warmup_update=warmup_update,
                update=0,
            )
            directory = checkpoints_dir / "warmup_last"
            save_checkpoint(directory, warmup_state, metadata)
            _save_runtime(directory, key, state, counters, daily_margin)
        # 720ステップ分の全局面を保持したままPPO本体へ進むとGPUメモリを圧迫する。
        del rollout, critic_batch, metrics, before
        train_state = core.create_train_state(model, {"params": warmup_state.params}, ppo_config)
    elif train_state is None:
        train_state = core.create_train_state(model, variables, ppo_config)

    # best/step_*と同じくHydraの実行ディレクトリ配下の相対名として扱う
    # (to_absolute_pathでchdir前の元cwd基準にすると、実行のたびに共有される
    # 固定パスになってしまい、別runのpool対戦相手が混ざり込む)。
    pool_dir = checkpoints_dir / cfg.ppo.pool_dir
    for update in range(start_update + 1, cfg.ppo.total_updates + 1):
        key, rollout_key, update_key, opponent_key, member_key, seat_key, eval_key = (
            jax.random.split(key, 7)
        )
        started = time.perf_counter()
        pool_members = list_pool_members(pool_dir)
        if cfg.ppo.full_game_rounds <= 0:
            opponent = select_opponent(
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
                    daily_margin,
                )
            else:
                if opponent == "anchor":
                    opponent_variables = anchor_variables
                else:
                    member = pool_members[
                        int(jax.random.randint(member_key, (), 0, len(pool_members)))
                    ]
                    opponent_variables = load_opponent_variables(member, train_state)
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
                    daily_margin,
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
                reference_variables["params"],
            )
            jax.block_until_ready(metrics.loss)
            state, counters, daily_margin = (
                rollout.final_state,
                rollout.final_counters,
                rollout.final_margin,
            )
            if update % cfg.ppo.log_interval == 0:
                elapsed = time.perf_counter() - started
                values = jax.device_get(metrics)
                invalid = float(jax.device_get(rollout.executor_invalid).mean())
                clamped = float(jax.device_get(rollout.executor_clamped).mean())
                invalid_unit = float(jax.device_get(rollout.executor_invalid_unit).mean())
                clamped_unit = float(jax.device_get(rollout.executor_clamped_unit).mean())
                dones = jax.device_get(rollout.dones)
                terminal_count = int(dones.sum())
                terminal_win = terminal_loss = float("nan")
                if opponent != "self" and terminal_count:
                    finished = jax.device_get(rollout.rewards)[..., learner_seat][dones]
                    terminal_win = float((finished > 0.5).mean())
                    terminal_loss = float((finished < -0.5).mean())
                daily_arr = jax.device_get(rollout.daily_rewards)
                rollout_steps = jax.device_get(rollout.states.step)
                daily_boundary = ((rollout_steps + 1) % cfg.rules.turns_per_day == 0) | dones
                daily_events = int(daily_boundary.sum())
                daily_abs_mean = float(abs(daily_arr[:, :, 0]).sum() / max(daily_events, 1))
                value_arr = jax.device_get(batch.old_value)
                return_arr = jax.device_get(batch.returns)
                advantage_arr = jax.device_get(batch.advantages)
                logger.info(
                    "update=%d seconds=%.1f samples/s=%.1f loss=%.4f policy_loss=%.4f "
                    "value_loss=%.4f entropy=%.4f reference_actor_l2=%.6f "
                    "approx_kl=%.4f clip_fraction=%.3f epochs=%d "
                    "terminal_count=%d terminal_win=%.3f terminal_loss=%.3f "
                    "daily_events=%d daily_abs_mean=%.4f "
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
                    values.reference_actor_l2,
                    values.approx_kl,
                    values.clip_fraction,
                    int(jax.device_get(epochs_completed)),
                    terminal_count,
                    terminal_win,
                    terminal_loss,
                    daily_events,
                    daily_abs_mean,
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

        else:
            round_opponents = []
            for _ in range(cfg.ppo.full_game_rounds):
                key, round_opponent_key, round_member_key = jax.random.split(key, 3)
                choice = select_opponent(
                    float(jax.random.uniform(round_opponent_key)),
                    bool(pool_members),
                    cfg.ppo.anchor_sample_prob,
                    cfg.ppo.pool_sample_prob,
                )
                pool_variables = None
                if choice == "pool":
                    member = pool_members[
                        int(jax.random.randint(round_member_key, (), 0, len(pool_members)))
                    ]
                    pool_variables = load_opponent_variables(member, train_state)
                round_opponents.append(
                    opponent_for_choice(
                        choice,
                        anchor_variables,
                        {"params": train_state.params},
                        pool_variables,
                    )
                )
            train_state, diagnostics = full_game.full_game_update(
                model,
                train_state,
                reference_variables["params"],
                replace(ppo_config, normalize_advantages=False),
                rollout_config,
                round_opponents,
                update_key,
                batch_size=cfg.env.batch_size,
                minibatch=cfg.ppo.minibatch_size,
                segment_length=cfg.ppo.full_game_segment_length,
                gamma=cfg.ppo.gamma,
                gae_lambda=cfg.ppo.gae_lambda,
                value_lambda=cfg.ppo.value_lambda,
            )
            if update % cfg.ppo.log_interval == 0:
                logger.info(
                    "update=%d seconds=%.1f games=%d loss=%.4f policy_loss=%.4f "
                    "value_loss=%.4f entropy=%.4f reference_actor_l2=%.6f "
                    "gradient_norm=%.4f round_cosine=%.3f win=%.3f lose=%.3f draw=%.3f "
                    "daily_abs_mean=%.4f value_mean=%.4f return_mean=%.4f "
                    "invalid=%.3f clamped=%.3f invalid_unit=%.3f clamped_unit=%.3f",
                    update,
                    time.perf_counter() - started,
                    diagnostics["games"],
                    diagnostics["loss"],
                    diagnostics["policy_loss"],
                    diagnostics["value_loss"],
                    diagnostics["entropy"],
                    diagnostics["reference_actor_l2"],
                    diagnostics["gradient_norm"],
                    diagnostics.get("round_gradient_cosine", float("nan")),
                    diagnostics["win"],
                    diagnostics["lose"],
                    diagnostics["draw"],
                    diagnostics["daily_abs_mean"],
                    diagnostics["value_mean"],
                    diagnostics["return_mean"],
                    diagnostics["invalid"],
                    diagnostics["clamped"],
                    diagnostics["invalid_unit"],
                    diagnostics["clamped_unit"],
                )
        metadata = _checkpoint_metadata(
            cfg, model_config, reference_path, phase="ppo", update=update
        )
        if cfg.ppo.eval_interval > 0 and update % cfg.ppo.eval_interval == 0:
            best_exists = (best_dir / "metadata.json").exists()
            best_variables = (
                load_opponent_variables(best_dir, train_state) if best_exists else anchor_variables
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
            best_summary = log_evaluation(
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
                anchor_summary = log_evaluation(update, "anchor", anchor_result)
            else:
                anchor_summary = best_summary
            promoted = (
                best_summary["win_rate"] >= cfg.ppo.promotion_win_rate
                and anchor_summary["win_rate"] >= cfg.ppo.anchor_promotion_win_rate
            )
            if promoted:
                save_checkpoint(best_dir, train_state, metadata)
                add_to_pool(pool_dir, train_state, metadata, cfg.ppo.pool_size)
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
            _save_runtime(directory, key, state, counters, daily_margin)
            prune_step_checkpoints(checkpoints_dir, cfg.ppo.keep_last_checkpoints)


if __name__ == "__main__":
    main()
