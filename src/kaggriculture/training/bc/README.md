# Behavior Cloning

固定slot方策をJAXで教師あり学習し、PPOの初期値を作ります。

```bash
uv run python -m kaggriculture.training.bc.train
```

初回だけリプレイJSONをCPUで読み、expert行動の正規化と固定shape `State + Intent` shardの作成を
行います。以後はshardを逐次先読みするため、全データをRAMへ保持せず、token化・合法mask・
Transformer・損失・更新をJAX/GPUで処理します。

```bash
# 選別manifest
uv run python -m kaggriculture.training.bc.train \
  data.selection_dir=data/training_sets/broad_winners

# smoke test
uv run python -m kaggriculture.training.bc.train \
  data.num_episodes=2 train.max_steps=2

# 全選別リプレイ
uv run python -m kaggriculture.training.bc.train \
  experiment.name=all_replays data.num_episodes=null
```

cacheは`data.cache_dir`へepisode単位で保存します。同じリプレイ・rulesなら実験間で再利用できます。
定期checkpointは`checkpoints/step_<n>`、validation lossが最小の重みは`checkpoints/best`です。

## Actorとcriticの共同学習

`value_loss_coefficient`を正値にすると、BCの方策lossと完了試合のreturnに対するvalue MSEを
一つのoptimizer更新で学習します。共有embedding/Transformerには両方の勾配が流れ、方策headには
BCだけ、privileged/macro/value branchにはvalue lossだけが流れます。モデル構造はPPOと共通です。

```bash
uv run python -m kaggriculture.training.bc.train \
  experiment.name=joint_bc_value \
  model.use_asymmetric_critic=true \
  train.init_value_checkpoint=outputs/value_pretrain/<run>/checkpoints/best \
  train.value_loss_coefficient=0.01 \
  train.daily_reward_coefficient=0.0 \
  data.value_cache_dir=data/cache/bc_value_terminal
```

方策教師とvalue教師は同じtrain/validation試合集合から独立にbatch化します。joint checkpointには
両方の重みと報酬設定を保存するため、PPOの`ppo.init_value_checkpoint`へ直接渡せます。

PPOへはcheckpointディレクトリを直接渡します。

```bash
uv run python -m kaggriculture.training.ppo.train \
  ppo.init_bc_checkpoint=/absolute/path/to/checkpoints/best
```
