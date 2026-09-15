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

PPOへはcheckpointディレクトリを直接渡します。

```bash
uv run python -m kaggriculture.training.ppo.train \
  ppo.init_bc_checkpoint=/absolute/path/to/checkpoints/best
```
