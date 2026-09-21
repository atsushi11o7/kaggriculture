# Critic事前学習

完了済みリプレイから、PPOと同じ「日次資産差改善＋終端cash勝敗」の割引累積報酬を作り、非対称criticを回帰します。criticは公開CLS・非公開情報に加え、現金・概算資産・土地・hand数の自分/相手/差分を専用MLPで受け取ります。既存BCのActor重みはoptimizerで固定され、再学習されません。`train.daily_reward_coefficient=0`なら終端勝敗だけを教師にできます。

```bash
uv run python -m kaggriculture.training.value_pretrain.train \
  train.actor_checkpoint=outputs/bc/broad_all_v1/2026-09-16/13-35-05/checkpoints/best \
  train.daily_reward_coefficient=0.05 \
  experiment.name=daily_terminal_value
```

`data.selection_dir`は任意です。指定しなければ`data.data_dir`以下の全episodeを使います。選別manifestはプレイヤー単位ですが、criticには両席の観測が必要なため、指定された試合を重複排除して両席とも使います。未完了試合は使いません。

PPOでは、保存された`checkpoints/best`をActorとcriticの初期値に指定します。固定BC相手は別途指定します。

```bash
uv run python -m kaggriculture.training.ppo.train \
  ppo.init_value_checkpoint=outputs/value_pretrain/daily_terminal_value/<日付>/<時刻>/checkpoints/best \
  ppo.anchor_bc_checkpoint=outputs/bc/broad_all_v1/2026-09-16/13-35-05/checkpoints/best \
  ppo.daily_reward_coefficient=0.05
```

通常は`gamma`、日次報酬の係数・scale・上限、モデル構造が事前学習時と一致しないcheckpointを拒否します。報酬を切り替える場合は、PPO開始前にActorを固定し、固定BCとの完了試合でcriticだけを再適応できます。

```bash
uv run python -m kaggriculture.training.ppo.train \
  ppo.init_value_checkpoint=outputs/value_pretrain/recent5_value_macro/2026-09-20/12-23-03/checkpoints/best \
  ppo.anchor_bc_checkpoint=/path/to/bc/checkpoints/best \
  ppo.daily_reward_coefficient=0 \
  ppo.rollout_horizon=720 \
  env.batch_size=32 \
  ppo.critic_warmup_updates=5
```

warm-upは各rolloutで終端まで観測できた局面だけをMonte Carlo教師に使います。Actorと参照方策は変更せず、`critic_macro_encoder`、`privileged_encoder`、`value_head`だけを更新します。終了後はwarm-up済みの重みから通常のPPOを開始します。途中状態は`checkpoints/warmup_last`へ保存され、`ppo.resume_checkpoint`で再開できます。
