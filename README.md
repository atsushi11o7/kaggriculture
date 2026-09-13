# kaggriculture

[Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) — Kaggleの2人対戦農業シムAIエージェントコンペ用リポジトリ。

## 概要

畑を管理し、種や家畜の購入・植え付け・収穫・市場での売買を行いながら、シーズン終了時の所持金を競う2人対戦シミュレーションゲーム。対戦エンジンは `kaggle-environments` の `kaggriculture` 環境を使用。


## 学習

推奨経路は、固定shape cacheを使うJAX BCで初期方策を作り、JAXシミュレータと接続した
PPOで自己対戦学習し、最後にActorだけをPyTorchへ変換する流れです。

```bash
# 学習用リプレイmanifestを作る
uv run python -m kaggriculture.training.replay_curation.select

# Behavior Cloning
uv run python -m kaggriculture.training.bc.train_jax \
  data.selection_dir=data/training_sets/broad_winners

# BC checkpointからPPO
uv run python -m kaggriculture.training.ppo.train \
  ppo.init_bc_checkpoint=/absolute/path/to/bc/checkpoints/best

# PPO checkpointを提出用PyTorch重みへ変換
uv run python -m kaggriculture.training.export_policy \
  /absolute/path/to/ppo/checkpoints/step_1000 submission/model_weights.pt
```

モデル構造、GPU/CPU境界、既知の近似、Hydraによる実験管理は
[`src/kaggriculture/policy/README.md`](src/kaggriculture/policy/README.md)、BCのデータcacheは
[`src/kaggriculture/training/bc/README.md`](src/kaggriculture/training/bc/README.md)を参照してください。
GBDT教師を使う任意経路は
[`src/kaggriculture/training/gbdt/README.md`](src/kaggriculture/training/gbdt/README.md)にあります。
