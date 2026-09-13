"""training.bc.trainのcheckpoint保存・再開ロジックの契約テスト。学習ループ全体
(Hydra起動)は`data/replays/`を使った手動のスモークテストで確認済みのため、ここでは
ネットワークに依存しない保存/読み込みの往復だけを検証する。
"""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from kaggriculture.policy.torch import model as M
from kaggriculture.training.bc.train import (
    _load_checkpoint,
    _load_initial_weights,
    _model_config,
    _prune_checkpoints,
    _save_checkpoint,
    _validate_config,
)
from tests.policy.conftest import DEFAULT_MODEL_CONFIG


def test_save_and_load_checkpoint_round_trips_weights_and_step(tmp_path: Path):
    torch.manual_seed(0)
    net = M.PolicyValueNet(DEFAULT_MODEL_CONFIG)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    # optimizer stateを実際に生成しておく(zero_grad直後は空でstate_dictが薄いため)。
    loss = sum(p.sum() for p in net.parameters())
    loss.backward()
    optimizer.step()

    _save_checkpoint(net, optimizer, step=7, checkpoint_dir=tmp_path)

    net2 = M.PolicyValueNet(DEFAULT_MODEL_CONFIG)
    optimizer2 = torch.optim.AdamW(net2.parameters(), lr=1e-3)
    step = _load_checkpoint(net2, optimizer2, tmp_path / "step_7.pt", device=torch.device("cpu"))

    assert step == 7
    for p1, p2 in zip(net.parameters(), net2.parameters(), strict=True):
        assert torch.equal(p1, p2)
    assert optimizer2.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()


def test_load_initial_weights_copies_weights_without_optimizer_state(tmp_path: Path):
    """_load_initial_weightsはfine-tuning用の重みだけの読み込みなので、
    optimizerには一切触れない(呼び出し側が新しいoptimizerをstep 0から使う前提)。"""
    torch.manual_seed(0)
    net = M.PolicyValueNet(DEFAULT_MODEL_CONFIG)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    loss = sum(p.sum() for p in net.parameters())
    loss.backward()
    optimizer.step()
    _save_checkpoint(net, optimizer, step=42, checkpoint_dir=tmp_path)

    torch.manual_seed(1)  # net2の初期重みがnetと異なることを保証する
    net2 = M.PolicyValueNet(DEFAULT_MODEL_CONFIG)
    optimizer2 = torch.optim.AdamW(net2.parameters(), lr=1e-3)

    _load_initial_weights(net2, tmp_path / "step_42.pt", device=torch.device("cpu"))

    for p1, p2 in zip(net.parameters(), net2.parameters(), strict=True):
        assert torch.equal(p1, p2)
    # 重みだけ読み込み、optimizer2自体には触れていない(空のまま)。
    assert optimizer2.state_dict()["state"] == {}


def test_prune_checkpoints_keeps_only_the_most_recent_by_step(tmp_path: Path):
    for step in [10, 30, 20, 40]:  # 保存順がstep順とは限らないことも確認する
        (tmp_path / f"step_{step}.pt").write_bytes(b"")

    _prune_checkpoints(tmp_path, keep_last=2)

    remaining = {p.name for p in tmp_path.glob("step_*.pt")}
    assert remaining == {"step_30.pt", "step_40.pt"}


def test_save_checkpoint_prunes_old_checkpoints_when_keep_last_is_set(tmp_path: Path):
    net = M.PolicyValueNet(DEFAULT_MODEL_CONFIG)
    optimizer = torch.optim.AdamW(net.parameters())

    for step in [1, 2, 3, 4]:
        _save_checkpoint(net, optimizer, step, tmp_path, keep_last=2)

    remaining = {p.name for p in tmp_path.glob("step_*.pt")}
    assert remaining == {"step_3.pt", "step_4.pt"}


def test_save_checkpoint_keeps_all_when_keep_last_is_zero(tmp_path: Path):
    net = M.PolicyValueNet(DEFAULT_MODEL_CONFIG)
    optimizer = torch.optim.AdamW(net.parameters())

    for step in [1, 2, 3]:
        _save_checkpoint(net, optimizer, step, tmp_path, keep_last=0)

    remaining = {p.name for p in tmp_path.glob("step_*.pt")}
    assert remaining == {"step_1.pt", "step_2.pt", "step_3.pt"}


def _valid_config():
    return OmegaConf.create(
        {
            "data": {
                "val_fraction": 0.1,
                "num_episodes": None,
                "manifest_dir": None,
                "min_avg_agent_score": None,
                "min_agent_score": None,
                "min_player_reward": None,
                "turns_per_day": 24,
                "shed_capacity": 100,
                "hire_mult": 1,
                "max_market_orders": 10,
            },
            "model": {
                "d_model": 128,
                "num_heads": 4,
                "d_feedforward": 512,
                "num_layers_encoder": 4,
                "num_layers_decoder": 3,
                "dropout": 0.0,
                "use_episode_history": False,
                "use_asymmetric_critic": False,
                "num_layers_critic": 2,
            },
            "train": {
                "batch_size": 64,
                "num_workers": 2,
                "max_epochs": 1,
                "log_interval": 10,
                "val_interval": 100,
                "val_batches": 20,
                "checkpoint_interval": 200,
                "lr": 3e-4,
                "weight_decay": 0.01,
                "grad_clip_norm": 1.0,
                "init_checkpoint": None,
                "init_weights": None,
            },
        }
    )


def test_validate_config_accepts_defaults_and_builds_model():
    cfg = _valid_config()

    _validate_config(cfg)
    net = M.PolicyValueNet(_model_config(cfg))

    assert net.token_embedding.bag.embedding_dim == 128


def test_validate_config_rejects_score_threshold_without_manifest():
    cfg = _valid_config()
    cfg.data.min_agent_score = 1000

    with pytest.raises(ValueError, match="manifest_dir"):
        _validate_config(cfg)


def test_validate_config_rejects_incompatible_attention_dimensions():
    cfg = _valid_config()
    cfg.model.d_model = 127

    with pytest.raises(ValueError, match="divisible"):
        _validate_config(cfg)


def test_checkpoint_stores_resolved_config(tmp_path: Path):
    net = M.PolicyValueNet(DEFAULT_MODEL_CONFIG)
    optimizer = torch.optim.AdamW(net.parameters())
    config = {"model": {"d_model": 128}}

    _save_checkpoint(net, optimizer, 3, tmp_path, config)

    checkpoint = torch.load(tmp_path / "step_3.pt", map_location="cpu")
    assert checkpoint["config"] == config
