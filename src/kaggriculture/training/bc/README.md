# JAX Behavior Cloning

新規BC学習はJAX版を使います。

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
JAX BC checkpointはPPOへ直接渡せます。

```bash
uv run python -m kaggriculture.training.ppo.train \
  ppo.init_bc_checkpoint=/absolute/path/to/checkpoints/best
```

PPO終了後は`training.export_policy`でActorだけをCPU PyTorch checkpointへ変換します。
旧`training.bc.train`はPyTorch checkpointとの互換用に残しています。

既定ハイパーパラメータは安全な開始点であり、最適値ではありません。特に
`train.learning_rate`、`train.batch_size`、PPOの`entropy_coef`と`temperature`は
実測比較してください。PPOの既定learning rateはBC重みを壊しにくいよう`1e-4`、評価は
32局を50 updateごとに実行します。ログ頻度を抑え、JAXの非同期実行を頻繁な
host同期で止めない設定です。
