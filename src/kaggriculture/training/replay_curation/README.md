# Replay curation

Kaggleリプレイをプレイヤー単位で解析し、再現可能なBC用manifestを作ります。元JSONは複製しません。

```bash
uv run python -m kaggriculture.training.replay_curation.select \
  experiment.name=broad_winners

uv run python -m kaggriculture.training.bc.train \
  data.selection_dir=data/training_sets/broad_winners
```

選別では終端所持金、勝敗margin、対局時の平均・最低Skill Ratingを利用できます。販売構成と
市場取引量から簡易戦略ラベルを付け、`max_per_strategy`で単一戦略への偏りを抑えます。
train/validation/testは試合単位で固定され、各JSONLには元ファイルのsizeとmtimeも記録されます。

`summary.json`には解決済みHydra設定とsplit・戦略別件数を保存します。同じ入力ファイルとseed、
設定からは同じmanifestが生成されます。


代表的なデータセットはHydra overrideだけで作り分けられます。閾値は実データの分布を
`summary.json`で確認して調整してください。

```bash
# 高Rating・高収益の勝者
uv run python -m kaggriculture.training.replay_curation.select \
  experiment.name=elite_winners \
  selection.min_agent_score=2900 \
  selection.min_terminal_cash=80000

# 戦略ごとの件数を制限して偏りを抑える
uv run python -m kaggriculture.training.replay_curation.select \
  experiment.name=diverse_winners \
  selection.max_per_strategy=1000
```

Rating CSVが持つのは対局全体の`avg_score`と`min_score`で、プレイヤー個別Ratingではありません。
そのためRatingは対局品質のフィルタ、`rewards[player]`は教師プレイヤー選択に使います。


戦略ラベルは`action.market`の**注文要求量**から作る粗いヒューリスティックで、実際に
約定した売上や作物生産量ではありません。データの多様性を調べる目安として使い、
強さの判定には終端所持金とclosed-loop評価を使ってください。
