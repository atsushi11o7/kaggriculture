"""BC(Behavior Cloning)学習ループ。

`uv run python -m kaggriculture.training.bc.train`で起動する。設定は
`training/conf/bc.yaml`(Hydra)。ログ・checkpointはHydraのrun dir(`outputs/`配下、
gitignore対象)にまとめる。
"""

import logging
import random
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.torch import distribution as D
from kaggriculture.policy.torch import model as M
from kaggriculture.rules import constants as C
from kaggriculture.training.bc.dataset import (
    ReplayActionDataset,
    filter_episodes_by_agent_score,
    list_episode_files,
    load_manifest,
    split_episode_files,
)
from kaggriculture.training.bc.objective import mean_token_nll

logger = logging.getLogger(__name__)


def _validate_config(cfg: DictConfig) -> None:
    """早い段階で、Hydra設定の矛盾と危険な値を検出する。"""
    if not 0 <= cfg.data.val_fraction < 1:
        raise ValueError("data.val_fraction must be in [0, 1)")
    if cfg.data.num_episodes is not None and cfg.data.num_episodes <= 0:
        raise ValueError("data.num_episodes must be positive or null")
    if cfg.data.turns_per_day <= 0 or cfg.data.shed_capacity <= 0:
        raise ValueError("turns_per_day and shed_capacity must be positive")
    if not 1 <= cfg.data.max_market_orders <= C.MAX_MARKET_ORDERS:
        raise ValueError(f"max_market_orders must be in [1, {C.MAX_MARKET_ORDERS}]")
    if cfg.data.hire_mult < 0:
        raise ValueError("data.hire_mult must be non-negative")
    if (
        cfg.data.min_avg_agent_score is not None or cfg.data.min_agent_score is not None
    ) and cfg.data.manifest_dir is None:
        raise ValueError("score thresholds require data.manifest_dir")
    _model_config(cfg)
    if cfg.model.use_episode_history:
        raise ValueError(
            "BC replay counters are not implemented; use_episode_history must be false"
        )
    if cfg.model.use_asymmetric_critic:
        raise ValueError("BC does not train a critic; use_asymmetric_critic must be false")
    for name in (
        "batch_size",
        "max_epochs",
        "log_interval",
        "val_interval",
        "val_batches",
        "checkpoint_interval",
    ):
        if cfg.train[name] <= 0:
            raise ValueError(f"train.{name} must be positive")
    if cfg.train.num_workers < 0:
        raise ValueError("train.num_workers must be non-negative")
    if cfg.train.lr <= 0 or cfg.train.weight_decay < 0 or cfg.train.grad_clip_norm <= 0:
        raise ValueError(
            "lr and grad_clip_norm must be positive; weight_decay must be non-negative"
        )
    if cfg.train.init_checkpoint is not None and cfg.train.init_weights is not None:
        raise ValueError("specify at most one of train.init_checkpoint / train.init_weights")


def _model_config(cfg: DictConfig) -> ModelConfig:
    """解決済みHydra model節をbackend共通の構成型へ変換する。"""
    values = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("model config must be a mapping")
    return ModelConfig(**values)


def _collate_pairs(batch: list[tuple[dict, dict]]) -> list[tuple[dict, dict]]:
    """(観測, 行動)は可変長の辞書なので、tensorへcollateせずそのまま束ねる。"""
    return batch


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_loader(dataset: ReplayActionDataset, cfg: DictConfig) -> DataLoader:
    # CUDA初期化後のforkは安全でないため、workerはspawnで起動する。
    multiprocessing_context = "spawn" if cfg.train.num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        collate_fn=_collate_pairs,
        multiprocessing_context=multiprocessing_context,
    )


def _run_validation(
    net: M.PolicyValueNet, val_loader: DataLoader, cfg: DictConfig, max_batches: int
) -> float | None:
    """val setの一部で平均token NLLを計算する。valが空ならNoneを返す。"""
    was_training = net.training
    net.eval()
    total_nll = 0.0
    total_decisions = 0
    data_kwargs = {
        "turns_per_day": cfg.data.turns_per_day,
        "shed_capacity": cfg.data.shed_capacity,
        "hire_mult": cfg.data.hire_mult,
        "max_market_orders": cfg.data.max_market_orders,
    }
    try:
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= max_batches:
                    break
                log_probs, _entropies, num_decisions = D.evaluate_policy_batch(
                    net, batch, **data_kwargs
                )
                total_nll += float(-log_probs.sum().item())
                total_decisions += sum(num_decisions)
    finally:
        net.train(was_training)
    if total_decisions == 0:
        return None
    return total_nll / total_decisions


def _save_checkpoint(
    net: M.PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: Path,
    config: dict | None = None,
    keep_last: int = 0,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"step_{step}.pt"
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )
    logger.info("saved checkpoint: %s", path)
    if keep_last > 0:
        _prune_checkpoints(checkpoint_dir, keep_last)


def _prune_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    """`step_<n>.pt`をstep番号順に並べ、直近keep_last件だけ残して残りを削除する。

    checkpointは1件あたりモデル+optimizer state(Adamはmomentを2つ持つため実質
    モデルの3倍程度)なので、長時間学習すると際限なく増える。checkpoint_interval
    ごとの保存はそのままに、ディスク使用量だけ抑える。
    """
    checkpoints = sorted(
        checkpoint_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.removeprefix("step_")),
    )
    for path in checkpoints[:-keep_last]:
        path.unlink()
        logger.info("removed old checkpoint: %s", path)


def _load_checkpoint(
    net: M.PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    path: Path,
    device: torch.device,
) -> int:
    """_save_checkpoint形式のcheckpointから重み・optimizer状態を読み込み、続きから
    学習するためのstepを返す。"""
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt["step"]
    logger.info("resumed from checkpoint: %s (step=%d)", path, step)
    return step


def _load_initial_weights(net: M.PolicyValueNet, path: Path, device: torch.device) -> None:
    """checkpointから重みだけを読み込む(optimizer state・stepは引き継がない)。

    再開(_load_checkpoint)とは別物で、別データセットでのfine-tuning用途を想定する
    (例: 大量データでBCした後、稼いだ額の多いエピソードだけに絞って続けて学習する)。
    optimizerのAdam moment推定は元の分布に対するものなので引き継がず、stepも0から
    数え直す。
    """
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    logger.info("loaded initial weights from: %s (checkpoint step=%s)", path, ckpt.get("step"))


@hydra.main(version_base=None, config_path="../conf", config_name="bc")
def main(cfg: DictConfig) -> None:
    _validate_config(cfg)
    torch.manual_seed(cfg.train.seed)
    device = _resolve_device(cfg.train.device)
    logger.info("device: %s", device)

    data_kwargs = {
        "turns_per_day": cfg.data.turns_per_day,
        "shed_capacity": cfg.data.shed_capacity,
        "hire_mult": cfg.data.hire_mult,
        "max_market_orders": cfg.data.max_market_orders,
    }
    cache_dir = (
        Path(to_absolute_path(cfg.data.cache_dir)) if cfg.data.cache_dir is not None else None
    )
    dataset_kwargs = {
        **data_kwargs,
        "min_player_reward": cfg.data.min_player_reward,
        "cache_dir": cache_dir,
        "validate_normalized": cfg.data.validate_normalized,
    }

    # hydra.job.chdir=trueによりcwdがrun dirへ変わるため、data_dirは元のcwd基準の
    # 絶対パスへ解決してから使う。
    data_dir = Path(to_absolute_path(cfg.data.data_dir))
    episode_files = list_episode_files(data_dir)
    if cfg.data.manifest_dir is not None:
        manifest_dir = Path(to_absolute_path(cfg.data.manifest_dir))
        manifest = load_manifest(manifest_dir)
        before = len(episode_files)
        episode_files = filter_episodes_by_agent_score(
            episode_files, manifest, cfg.data.min_avg_agent_score, cfg.data.min_agent_score
        )
        logger.info(
            "score filter (manifest=%s): %d -> %d episodes",
            manifest_dir,
            before,
            len(episode_files),
        )
    if cfg.data.num_episodes is not None:
        random.Random(cfg.data.split_seed).shuffle(episode_files)
        episode_files = episode_files[: cfg.data.num_episodes]
    if not episode_files:
        raise ValueError(f"no episode files found under {data_dir} (after filtering)")
    train_files, val_files = split_episode_files(
        episode_files, cfg.data.val_fraction, cfg.data.split_seed
    )
    if not train_files or (cfg.data.val_fraction > 0 and not val_files):
        raise ValueError("not enough episodes for the requested train/validation split")
    logger.info("episodes: %d train, %d val", len(train_files), len(val_files))

    val_loader = _make_loader(ReplayActionDataset(val_files, **dataset_kwargs), cfg)

    model_config = _model_config(cfg)
    net = M.PolicyValueNet(model_config).to(device)
    if cfg.train.init_weights is not None:
        init_weights = Path(to_absolute_path(cfg.train.init_weights))
        _load_initial_weights(net, init_weights, device)

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    writer = SummaryWriter(log_dir=".")
    checkpoint_dir = Path("checkpoints")

    step = 0
    if cfg.train.init_checkpoint is not None:
        init_checkpoint = Path(to_absolute_path(cfg.train.init_checkpoint))
        step = _load_checkpoint(net, optimizer, init_checkpoint, device)

    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    initial_step = step
    for epoch in range(cfg.train.max_epochs):
        epoch_files = list(train_files)
        random.Random(cfg.train.seed + epoch).shuffle(epoch_files)
        train_loader = _make_loader(ReplayActionDataset(epoch_files, **dataset_kwargs), cfg)
        for batch in train_loader:
            if cfg.train.max_steps > 0 and step >= cfg.train.max_steps:
                break
            net.train()
            optimizer.zero_grad(set_to_none=True)
            log_probs, _entropies, num_decisions = D.evaluate_policy_batch(
                net, batch, **data_kwargs
            )
            loss = mean_token_nll(log_probs, num_decisions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip_norm)
            optimizer.step()
            step += 1

            if step % cfg.train.log_interval == 0:
                logger.info("epoch %d step %d train_loss %.4f", epoch, step, loss.item())
                writer.add_scalar("train/loss", loss.item(), step)

            if step % cfg.train.val_interval == 0:
                val_loss = _run_validation(net, val_loader, cfg, cfg.train.val_batches)
                if val_loss is not None:
                    logger.info("epoch %d step %d val_loss %.4f", epoch, step, val_loss)
                    writer.add_scalar("val/loss", val_loss, step)

            if step % cfg.train.checkpoint_interval == 0:
                _save_checkpoint(
                    net,
                    optimizer,
                    step,
                    checkpoint_dir,
                    resolved_config,
                    cfg.train.keep_last_checkpoints,
                )

            if cfg.train.max_steps > 0 and step >= cfg.train.max_steps:
                break
        if cfg.train.max_steps > 0 and step >= cfg.train.max_steps:
            break

    already_at_target = cfg.train.max_steps > 0 and initial_step >= cfg.train.max_steps
    if step == initial_step and not already_at_target:
        writer.close()
        raise ValueError("no training samples remained after applying data filters")
    if step == 0 or step % cfg.train.checkpoint_interval != 0:
        _save_checkpoint(
            net, optimizer, step, checkpoint_dir, resolved_config, cfg.train.keep_last_checkpoints
        )
    writer.close()


if __name__ == "__main__":
    main()
