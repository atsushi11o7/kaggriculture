# GBDT teacher

GBDTは最終提出モデルではなく、少量の高品質リプレイから候補順位を学び、新しい教師対局を
生成する中間方策です。Transformerの構造や既存BC/PPOの損失は変更しません。

```text
Kaggle replay
  -> LightGBM LambdaRank
  -> GBDT teacher matches
  -> existing JAX BC
  -> existing JAX PPO
  -> PyTorch actor export
```

## 1. GBDTを学習する

```bash
uv run python -m kaggriculture.training.gbdt.train \
  experiment.name=high_skill \
  data.manifest_dir=data/replays/manifests \
  data.min_agent_score=2000
```

各decisionをquery groupとし、合法候補を行、expertが選んだ候補をrelevance 1として
`unit_op`、`market_op`、`quantity`の3 rankerを学習します。分割はターン単位ではなく
試合単位です。データ条件とゲーム規則からcache名を決めるため、条件を変えても古いcacheを
誤利用しません。候補行は種別ごとのバイナリへ逐次保存し、学習時はmemmapと
LightGBM `Sequence`から読みます。全行をNumPy配列へ連結しません。
変換の正常件数・破棄件数・理由別件数は、実験出力の`data_summary.json`へ保存します。

数量候補は学習時だけ既定16件へ削減します。expert数量、1、最大数量、expert近傍、
対数間隔の代表値を必ず含む決定論的な負例選択です。推論時の候補は削減せず、
従来どおり合法な全数量を採点します。全件が既定なので、小規模確認には
`data.num_episodes=32`などを明示してください。cacheは生特徴量のためディスク容量を使います。

GBDTの表形式特徴には現在の局面、候補のop/item/quantity、移動先タイル、対象在庫、価格を
含めます。これはTransformerの埋め込みとは独立した教師用特徴です。

### market_opのquery重み付け

`market_op`はSTOP/HIREだけで全決定の過半数を占め、素のLambdaRankだと比較勾配がそこへ
支配され、BUY_SEED/BUY_LAND/BUY_ANIMAL/BUY_PRODUCTのような出現頻度は低いが決定的に
重要な行動をほぼ選べなくなります(実際に自己対戦で種を一度も購入せず資金が尽きる問題を
確認しました)。そのため`build_ranking_files`/`build_ranking_data`は、正解opの出現頻度の
平方根に反比例するquery重みを計算し(完全な逆頻度だと出現数十件のクラスが極端な重みに
なるため`[0.25, 8.0]`にクリップし、平均が1.0になるよう正規化)、同じqueryに属する全候補
行へ同じ重みを与えます。`unit_op`は対象外です(移動方向の低精度は多数派崩壊ではなく
位置依存の難しさが原因)。`quantity`も対象外です(離散クラスを持たない)。

学習後、`market_op`だけクラス別top1精度・macro精度・予測分布を`class_report.json`へ
保存し、ログにも出力します。全体top1はSTOP/HIREの高精度に隠れるため、少数派クラスの
実際の性能はこちらで確認してください。

現在のPyPI LightGBM wheelにはCUDA learnerが含まれないため、GBDT学習と推論はCPUです。
`model.n_jobs=4`で全コア占有を避けます。大量計算の中心となる蒸留BCとPPOはJAX/GPUで
実行されます。小さな候補集合を逐次評価するGBDT推論は、GPU転送の恩恵が出にくい処理です。

## 2. 教師対局を生成する

```bash
uv run python -m kaggriculture.training.gbdt.generate \
  model.checkpoint=/absolute/path/to/outputs/gbdt/high_skill/.../checkpoint \
  num_games=1000
```

GBDTは`policy.torch.candidate_api`の公開契約を通じて、Transformer方策と同じ合法候補・
shadow state更新・決定順を使います。方策内部のgeneratorや状態型には直接依存しません。
`temperature=0`はgreedy、正値は候補スコアからサンプリングします。既定値は少量の探索を
入れる`0.15`です。

同一GBDT同士だけの対局は分布が狭くなります。本格運用では、元の強いルールベース方策や
異なるGBDT checkpointも相手へ混ぜ、生成データのclosed-loop成績を確認してください。

## 3. 既存BCへ渡す

生成ファイルはKaggle episode JSONなので、既存JAX BCへそのまま渡せます。

```bash
uv run python -m kaggriculture.training.bc.train \
  experiment.name=gbdt_distillation \
  data.data_dir=data/replays/gbdt-generated \
  data.num_episodes=null
```

この段階がGBDTからTransformerへのBC（蒸留）です。出力されるTransformer checkpointは
通常のJAX BC checkpointなので、そのままPPOへ渡せます。

```bash
uv run python -m kaggriculture.training.ppo.train \
  ppo.init_bc_checkpoint=/absolute/path/to/checkpoints/best
```

GBDT checkpointをPPOへ直接渡すことはできません。決定木とTransformerでは重み構造が異なる
ため、必ず教師対局の生成とBCを経由します。
