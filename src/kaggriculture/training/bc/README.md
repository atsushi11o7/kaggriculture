# Behavior Cloning

## 推奨: JAX BC

新規BC学習はJAX版を使います。Kaggleリプレイを勝者・Skill Rating・戦略多様性で
選別する場合は、先に`training/replay_curation/README.md`のmanifestを作成します。

```bash
uv run python -m kaggriculture.training.bc.train_jax
```

初回実行ではリプレイJSONを読み、expert行動の正規化と合法候補への照合をCPUで一度だけ
行います。その結果を`State + player + choices + decision_mask`の固定shape圧縮shardとして
`data.cache_dir`へ保存します。

2回目以降はshardを先読みし、観測token化、合法手mask、教師強制Transformer、NLL、勾配更新を
JAX上で実行します。同じリプレイとゲーム規則なら、複数epochやHydra実験間でもcacheを再利用
できます。

既定値は全日付からseed固定で256試合を抽出し、2 epoch学習します。手元の全リプレイを
使う場合は`data.num_episodes=null`を指定します。生JSON全体は大きいため、まず既定値で
cache生成時間、GPUメモリ、closed-loop成績を確認してから広げてください。

```bash
# 選別済みmanifestを使う
uv run python -m kaggriculture.training.bc.train_jax \
  data.selection_dir=data/training_sets/broad_winners

# 小規模な動作確認
uv run python -m kaggriculture.training.bc.train_jax \
  data.num_episodes=2 train.max_steps=2

# 高Skill Rating対局へ絞る例
uv run python -m kaggriculture.training.bc.train_jax \
  experiment.name=high_skill \
  data.manifest_dir=data/replays/manifests \
  data.min_agent_score=2000 \
  train.max_epochs=3

# 全リプレイを使う本学習
uv run python -m kaggriculture.training.bc.train_jax \
  experiment.name=all_replays \
  data.num_episodes=null
```

validation lossが最小の重みは`checkpoints/best`、定期保存は`checkpoints/step_<n>`へ入ります。
最良lossもcheckpointへ保存されるため、中断再開後に悪い重みで`best`を上書きしません。
JAX BC checkpointはPPOへ直接渡せます。

Cache作成時の正常件数・破棄件数・理由別件数は、実験出力の`data_summary.json`と
各shardの`.stats.json`へ保存します。破棄率が増えた場合は、ルール変更や
リプレイschemaの差分を確認してください。

```bash
uv run python -m kaggriculture.training.ppo.train \
  ppo.init_bc_checkpoint=/absolute/path/to/checkpoints/best
```

PPO終了後は`training.export_policy`でActorだけをCPU PyTorch checkpointへ変換します。

## 代替: PyTorch BC

`python -m kaggriculture.training.bc.train`も現役の代替経路として利用できます。
生リプレイを`ReplayActionDataset`で逐次正規化し、PyTorchモデルを直接学習します。
既存PyTorch checkpointの再開、Torch/JAX比較、JAX経路の切り分けに有用です。

大量データを複数回学習する通常運用では、固定shape cacheを再利用でき、そのまま
JAX PPOへ接続できるJAX BCを推奨します。提出モデルがPyTorchであることは、
PyTorch BCを使う理由にはなりません。提出時の変換は`training.export_policy`が担当します。

既定ハイパーパラメータは安全な開始点であり、最適値ではありません。特に
`train.learning_rate`、`train.batch_size`、PPOの`entropy_coef`と`temperature`は
実測比較してください。PPOの既定learning rateはBC重みを壊しにくいよう`1e-4`、評価は各相手について両席32局ずつ（計64局）を50 updateごとに実行します。ログ頻度を抑え、JAXの非同期実行を頻繁な
host同期で止めない設定です。
