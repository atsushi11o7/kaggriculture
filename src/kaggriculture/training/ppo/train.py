"""固定slot方策専用のGPU完結PPO学習入口。"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from flax import serialization, traverse_util
from flax.core import freeze, unfreeze
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from kaggriculture.policy.common.config import (
    CRITIC_PARAMETER_MODULES,
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
from kaggriculture.training.ppo import core
from kaggriculture.training.ppo.evaluation import evaluate_both_seats
from kaggriculture.training.ppo.rollout import (
    RolloutConfig,
    collect_rollout,
    collect_rollout_vs_opponent,
    to_critic_batch,
    to_ppo_batch,
)

logger = logging.getLogger(__name__)


def _validate_critic_architecture(metadata: dict, path: Path) -> None:
    """checkpointに記録されたcritic実装とversionの組み合わせを検証する。"""
    variant = metadata.get("model_variant", "shared")
    if metadata.get("critic_architecture_version") != critic_version(variant):
        raise ValueError(f"incompatible critic architecture: {path}")


_ACTOR_CONFIG_FIELDS = (
    "d_model",
    "num_heads",
    "d_feedforward",
    "num_layers_encoder",
    "num_layers_decoder",
    "dropout",
    "use_episode_history",
)


def _load_actor_checkpoint(path: Path, variables: dict, target_config: ModelConfig) -> dict:
    """互換BC/PPO checkpointのActor重みを現在のモデルへ読み込む。

    Args:
        path: checkpointディレクトリ。
        variables: criticを含む現在のモデルの初期variables。
        target_config: 現在のモデル設定。

    Returns:
        Actorを復元し、criticを初期値のまま残したvariables。

    Raises:
        ValueError: Actorのパラメータ名または形状が一致しない場合。
    """
    metadata = read_checkpoint_metadata(path)
    validate_checkpoint_metadata(metadata)
    source_config = ModelConfig(**metadata["model_config"])
    if any(
        getattr(source_config, field) != getattr(target_config, field)
        for field in _ACTOR_CONFIG_FIELDS
    ):
        raise ValueError(f"incompatible actor checkpoint config: {path}")
    saved = serialization.msgpack_restore((path / "state.msgpack").read_bytes())
    source = traverse_util.flatten_dict(saved["params"])
    target = traverse_util.flatten_dict(unfreeze(variables["params"]))

    def actor(name):
        return name[0] not in CRITIC_PARAMETER_MODULES

    source_actor = {name for name in source if actor(name)}
    target_actor = {name for name in target if actor(name)}
    if source_actor != target_actor or any(
        source[name].shape != target[name].shape for name in source_actor & target_actor
    ):
        raise ValueError(f"incompatible actor checkpoint: {path}")
    for name in target_actor:
        target[name] = jnp.asarray(source[name])

    # 分離版のcritic公開Encoderは、Actor checkpointの公開Encoderを複製して始める。
    # value head等は初期値のままなので、actor-only loadの意味は維持される。
    aliases = {
        "critic_token_embedding": "token_embedding",
        "critic_board_position_embedding": "board_position_embedding",
        "critic_encoder": "encoder",
    }
    for name in target:
        if name[0] not in aliases:
            continue
        source_name = (aliases[name[0]], *name[1:])
        if source_name not in source or source[source_name].shape != target[name].shape:
            raise ValueError(f"incompatible separated critic initialization: {path}")
        target[name] = jnp.asarray(source[source_name])
    return {"params": freeze(traverse_util.unflatten_dict(target))}


def _load_value_checkpoint(
    path: Path,
    variables: dict,
    target_config: ModelConfig,
    gamma: float,
    daily_reward_coefficient: float,
    daily_reward_scale: float,
    daily_reward_maximum: float,
    *,
    allow_reward_mismatch: bool = False,
) -> dict:
    """事前学習済みActorとcriticをPPOへ読み込む。

    報酬不一致は、直後にcritic-only warm-upで再適応する場合だけ許可する。
    """
    metadata = read_checkpoint_metadata(path)
    validate_checkpoint_metadata(metadata)
    is_value_checkpoint = metadata.get("trainer") == "value_pretrain" or (
        metadata.get("trainer") == "bc" and metadata.get("joint_value_training") is True
    )
    if not is_value_checkpoint:
        raise ValueError(f"not a value-pretraining checkpoint: {path}")
    _validate_critic_architecture(metadata, path)
    reward_mode = metadata.get("reward_mode")
    if reward_mode == "terminal_win":
        source_reward = (0.0, 10000.0, 0.02)
    elif reward_mode == "terminal_win_daily_asset":
        source_reward = (
            float(metadata["daily_reward_coefficient"]),
            float(metadata["daily_reward_scale"]),
            float(metadata["daily_reward_maximum"]),
        )
    else:
        raise ValueError(f"unsupported value checkpoint reward: {path}")
    target_reward = (
        daily_reward_coefficient,
        daily_reward_scale,
        daily_reward_maximum,
    )
    if abs(float(metadata["gamma"]) - gamma) > 1e-9:
        raise ValueError("value checkpoint gamma differs from PPO")
    reward_mismatch = any(
        abs(source - target) > 1e-9
        for source, target in zip(source_reward, target_reward, strict=True)
    )
    if reward_mismatch and not allow_reward_mismatch:
        raise ValueError("value checkpoint reward configuration differs from PPO")
    if ModelConfig(**metadata["model_config"]) != target_config:
        raise ValueError(f"incompatible value checkpoint model config: {path}")
    saved = serialization.msgpack_restore((path / "state.msgpack").read_bytes())
    source = traverse_util.flatten_dict(saved["params"])
    target = traverse_util.flatten_dict(unfreeze(variables["params"]))
    aliases = {
        "critic_token_embedding": "token_embedding",
        "critic_board_position_embedding": "board_position_embedding",
        "critic_encoder": "encoder",
    }
    restored = {}
    for name, target_value in target.items():
        source_name = name
        if source_name not in source and name[0] in aliases:
            source_name = (aliases[name[0]], *name[1:])
        if source_name not in source or source[source_name].shape != target_value.shape:
            raise ValueError(f"incompatible value checkpoint parameters: {path}")
        restored[name] = jnp.asarray(source[source_name])
    return {"params": freeze(traverse_util.unflatten_dict(restored))}


def _opponent_variables(directory: Path, train_state) -> dict:
    """train_stateと同じ構造をtemplateにして、checkpointのparamsだけを取り出す。"""
    metadata = read_checkpoint_metadata(directory)
    validate_checkpoint_metadata(metadata)
    _validate_critic_architecture(metadata, directory)
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
        "candidate_pass_rate": float(jax.device_get(result.pass_rate).mean()),
        "opponent_pass_rate": float(jax.device_get(result.opponent_pass_rate).mean()),
    }


def _log_evaluation(update: int, opponent: str, result) -> dict[str, float | int]:
    summary = _evaluation_summary(result)
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
        daily_reward_coefficient=cfg.ppo.daily_reward_coefficient,
        daily_reward_scale=cfg.ppo.daily_reward_scale,
        daily_reward_maximum=cfg.ppo.daily_reward_maximum,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="ppo")
def main(cfg: DictConfig) -> None:
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
    model_config = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    model = create_model(model_config, cfg.ppo.model_variant)
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
    anchor_variables = _load_actor_checkpoint(anchor_path, initial_variables, model_config)
    if cfg.ppo.init_bc_checkpoint:
        variables = _load_actor_checkpoint(
            Path(to_absolute_path(cfg.ppo.init_bc_checkpoint)), initial_variables, model_config
        )
    if cfg.ppo.init_value_checkpoint:
        variables = _load_value_checkpoint(
            Path(to_absolute_path(cfg.ppo.init_value_checkpoint)),
            initial_variables,
            model_config,
            cfg.ppo.gamma,
            cfg.ppo.daily_reward_coefficient,
            cfg.ppo.daily_reward_scale,
            cfg.ppo.daily_reward_maximum,
            allow_reward_mismatch=cfg.ppo.critic_warmup_updates > 0,
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
        else _load_actor_checkpoint(reference_path, initial_variables, model_config)
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
        _validate_critic_architecture(metadata, directory)
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
            metadata = {
                **checkpoint_shape_metadata(),
                "trainer": "ppo",
                "model_variant": cfg.ppo.model_variant,
                "phase": "critic_warmup",
                "critic_architecture_version": critic_version(cfg.ppo.model_variant),
                "warmup_update": warmup_update,
                "update": 0,
                "model_config": asdict(model_config),
                "reference_checkpoint_resolved": str(reference_path.resolve()),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            directory = checkpoints_dir / "warmup_last"
            save_checkpoint(directory, warmup_state, metadata)
            save_pytree(
                directory / "runtime.msgpack",
                {
                    "key": key,
                    "state": state,
                    "counters": counters,
                    "daily_margin": daily_margin,
                },
            )
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
                daily_margin,
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

        metadata = {
            **checkpoint_shape_metadata(),
            "trainer": "ppo",
            "model_variant": cfg.ppo.model_variant,
            "phase": "ppo",
            "critic_architecture_version": critic_version(cfg.ppo.model_variant),
            "update": update,
            "model_config": asdict(model_config),
            "reference_checkpoint_resolved": str(reference_path.resolve()),
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
                directory / "runtime.msgpack",
                {"key": key, "state": state, "counters": counters, "daily_margin": daily_margin},
            )
            _prune_step_checkpoints(checkpoints_dir, cfg.ppo.keep_last_checkpoints)


if __name__ == "__main__":
    main()
