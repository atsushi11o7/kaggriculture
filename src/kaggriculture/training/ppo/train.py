"""JAX end-to-end PPO学習入口。

`uv run python -m kaggriculture.training.ppo.train`で起動する。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import hydra
import jax
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import distribution as D
from kaggriculture.policy.jax import history as H
from kaggriculture.policy.jax import model as JM
from kaggriculture.simulator.reset import reset
from kaggriculture.training.checkpoint import load_checkpoint, save_checkpoint
from kaggriculture.training.ppo import core
from kaggriculture.training.ppo.evaluation import evaluate_closed_loop
from kaggriculture.training.ppo.rollout import RolloutConfig, collect_rollout, to_ppo_batch

logger = logging.getLogger(__name__)


def _validate_config(cfg: DictConfig) -> None:
    if cfg.env.batch_size <= 0 or cfg.ppo.rollout_horizon <= 0:
        raise ValueError("batch_size and rollout_horizon must be positive")
    if cfg.env.board_size <= 0 or cfg.env.board_size % 2:
        raise ValueError("board_size must be a positive even number")
    for name in ("turns_per_day", "shed_capacity", "episode_steps"):
        if cfg.env[name] <= 0:
            raise ValueError(f"env.{name} must be positive")
    if not 0 <= cfg.env.weed_chance <= 1 or cfg.env.hire_mult < 0:
        raise ValueError("weed_chance must be in [0, 1] and hire_mult non-negative")
    _model_config(cfg)

    sample_size = cfg.env.batch_size * cfg.ppo.rollout_horizon * 2
    if cfg.ppo.minibatch_size <= 0 or sample_size % cfg.ppo.minibatch_size:
        raise ValueError("minibatch_size must divide batch_size * rollout_horizon * 2")
    for name in ("update_epochs", "total_updates", "log_interval", "checkpoint_interval"):
        if cfg.ppo[name] <= 0:
            raise ValueError(f"ppo.{name} must be positive")
    if cfg.ppo.learning_rate <= 0 or cfg.ppo.weight_decay < 0 or cfg.ppo.max_grad_norm <= 0:
        raise ValueError(
            "learning_rate/max_grad_norm must be positive and weight_decay non-negative"
        )
    if not 0 <= cfg.ppo.gamma <= 1 or not 0 <= cfg.ppo.gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    if cfg.ppo.clip_epsilon <= 0 or cfg.ppo.value_clip_epsilon <= 0:
        raise ValueError("PPO clip epsilons must be positive")
    if cfg.ppo.value_coef < 0 or cfg.ppo.entropy_coef < 0 or cfg.ppo.temperature <= 0:
        raise ValueError("loss coefficients must be non-negative and temperature positive")
    if cfg.ppo.ratio_mode not in ("token", "joint"):
        raise ValueError("ppo.ratio_mode must be token or joint")
    if cfg.ppo.eval_interval < 0 or cfg.ppo.eval_episodes <= 0:
        raise ValueError("eval_interval must be non-negative and eval_episodes positive")
    if cfg.ppo.init_bc_checkpoint is not None and cfg.ppo.resume_checkpoint is not None:
        raise ValueError("init_bc_checkpoint and resume_checkpoint are mutually exclusive")


def _model_config(cfg: DictConfig) -> ModelConfig:
    values = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("model config must be a mapping")
    return ModelConfig(**values)


def _ppo_config(cfg: DictConfig) -> core.PPOConfig:
    names = set(core.PPOConfig.__dataclass_fields__)
    rule_values = {
        "turns_per_day": cfg.env.turns_per_day,
        "shed_capacity": cfg.env.shed_capacity,
        "hire_mult": cfg.env.hire_mult,
    }
    values = {name: cfg.ppo[name] for name in names - set(rule_values)}
    return core.PPOConfig(**values, **rule_values)


def _rollout_config(cfg: DictConfig) -> RolloutConfig:
    env = OmegaConf.to_container(cfg.env, resolve=True)
    if not isinstance(env, dict):
        raise TypeError("env config must be a mapping")
    env.pop("batch_size")
    return RolloutConfig(horizon=cfg.ppo.rollout_horizon, temperature=cfg.ppo.temperature, **env)


def _load_torch_bc_actor(path: Path, target: dict, config: ModelConfig) -> dict:
    """PyTorch BC checkpointのactorをJAX variablesへ読み込む。"""
    import torch

    from kaggriculture.policy.torch import model as TM
    from kaggriculture.training.weight_bridge import merge_actor_variables, torch_to_jax

    actor_args = {
        "d_model": config.d_model,
        "num_heads": config.num_heads,
        "d_feedforward": config.d_feedforward,
        "num_layers_encoder": config.num_layers_encoder,
        "num_layers_decoder": config.num_layers_decoder,
        "dropout": config.dropout,
        "use_episode_history": config.use_episode_history,
        "use_asymmetric_critic": False,
        "num_layers_critic": config.num_layers_critic,
    }
    net = TM.PolicyValueNet(ModelConfig(**actor_args))
    checkpoint = torch.load(path, map_location="cpu")
    net.load_state_dict(checkpoint["model_state_dict"])
    source = torch_to_jax(net, replace(config, use_asymmetric_critic=False))
    return merge_actor_variables(source, target) if config.use_asymmetric_critic else source


def _load_jax_bc_actor(path: Path, target: dict, config: ModelConfig) -> dict:
    """JAX BC checkpointのactorをPPOモデルへ直接読み込む。"""
    from kaggriculture.training.bc.jax_core import BCConfig, create_train_state
    from kaggriculture.training.weight_bridge import merge_actor_variables

    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    source_config = ModelConfig(**metadata["model_config"])
    comparable = (
        "d_model",
        "num_heads",
        "d_feedforward",
        "num_layers_encoder",
        "num_layers_decoder",
        "use_episode_history",
    )
    if any(getattr(source_config, name) != getattr(config, name) for name in comparable):
        raise ValueError("JAX BC actor structure does not match PPO model")
    source_model = JM.PolicyValueNet(source_config)
    source_variables = JM.initialize(source_model, jax.random.key(0))
    source_state = create_train_state(
        source_model, source_variables, BCConfig(**metadata["bc_config"])
    )
    restored, _ = load_checkpoint(path, source_state)
    return merge_actor_variables({"params": restored.params}, target)


def _load_bc_actor(path: Path, target: dict, config: ModelConfig) -> dict:
    """JAX BC directoryまたは旧PyTorch BCファイルからactorを読み込む。"""
    if path.is_dir():
        return _load_jax_bc_actor(path, target, config)
    return _load_torch_bc_actor(path, target, config)


def _prune_checkpoints(directory: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(
        directory.glob("step_*"), key=lambda path: int(path.name.removeprefix("step_"))
    )
    for path in checkpoints[:-keep_last]:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()


@hydra.main(version_base=None, config_path="../conf", config_name="ppo")
def main(cfg: DictConfig) -> None:
    _validate_config(cfg)
    model_config = _model_config(cfg)
    ppo_config = _ppo_config(cfg)
    rollout_config = _rollout_config(cfg)
    model = JM.PolicyValueNet(model_config)

    key = jax.random.key(cfg.ppo.seed)
    key, init_key, reset_key, cache_key, eval_cache_key = jax.random.split(key, 5)
    variables = JM.initialize(model, init_key)
    if cfg.ppo.init_bc_checkpoint is not None:
        path = Path(to_absolute_path(cfg.ppo.init_bc_checkpoint))
        variables = _load_bc_actor(path, variables, model_config)
        logger.info("loaded BC actor: %s", path)
    train_state = core.create_train_state(model, variables, ppo_config)
    start_update = 0
    if cfg.ppo.resume_checkpoint is not None:
        path = Path(to_absolute_path(cfg.ppo.resume_checkpoint))
        train_state, metadata = load_checkpoint(path, train_state)
        start_update = int(metadata["update"])
        logger.info("resumed JAX PPO checkpoint: %s (update=%d)", path, start_update)

    reference_variables = {"params": train_state.params}
    eval_cache = None
    if cfg.ppo.eval_interval > 0:
        eval_cache = D.init_decode_cache(model, eval_cache_key, cfg.ppo.eval_episodes)

    state = reset(
        reset_key,
        cfg.env.batch_size,
        board_size=cfg.env.board_size,
        starting_money=cfg.env.starting_money,
    )
    counters = H.zeros(cfg.env.batch_size)
    cache = D.init_decode_cache(model, cache_key, cfg.env.batch_size * 2)
    checkpoint_root = Path("checkpoints")
    resolved = OmegaConf.to_container(cfg, resolve=True)

    for update in range(start_update + 1, cfg.ppo.total_updates + 1):
        key, rollout_key, update_key = jax.random.split(key, 3)
        variables = {"params": train_state.params}
        rollout = collect_rollout(
            model, variables, cache, rollout_config, state, counters, rollout_key
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
        state = rollout.final_state
        counters = rollout.final_counters

        if update % cfg.ppo.log_interval == 0:
            scalars = jax.device_get(metrics)
            logger.info(
                "update=%d loss=%.4f policy=%.4f value=%.4f entropy=%.4f kl=%.5f clip=%.3f",
                update,
                scalars.loss,
                scalars.policy_loss,
                scalars.value_loss,
                scalars.entropy,
                scalars.approx_kl,
                scalars.clip_fraction,
            )
        if cfg.ppo.eval_interval > 0 and update % cfg.ppo.eval_interval == 0:
            key, eval_key = jax.random.split(key)
            evaluation = evaluate_closed_loop(
                model,
                {"params": train_state.params},
                reference_variables,
                eval_cache,
                replace(rollout_config, temperature=0.5),
                eval_key,
                cfg.ppo.eval_episodes,
            )
            result = jax.device_get(evaluation)
            logger.info(
                "closed_loop update=%d win_rate=%.3f cash=%.1f opponent_cash=%.1f",
                update,
                result.win_rate,
                result.cash[:, 0].mean(),
                result.cash[:, 1].mean(),
            )

        if update % cfg.ppo.checkpoint_interval == 0 or update == cfg.ppo.total_updates:
            directory = checkpoint_root / f"step_{update}"
            save_checkpoint(
                directory,
                train_state,
                {
                    "update": update,
                    "model_config": asdict(model_config),
                    "ppo_config": asdict(ppo_config),
                    "config": resolved,
                },
            )
            _prune_checkpoints(checkpoint_root, cfg.ppo.keep_last_checkpoints)
            logger.info("saved checkpoint: %s", directory)


if __name__ == "__main__":
    main()
