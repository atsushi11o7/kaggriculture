# kaggriculture GPU simulator

`kaggle-environments`の`kaggriculture`環境を、JAXでバッチ実行できるように移植したもの。RLの学習ループで大量のロールアウトを高速に集めるための実行エンジンで、**大会への提出物ではない**(提出はKaggle提供版のインターフェースに合わせて別途書く)。

行動仕様の詳細(op一覧・パラメータの意味など)は元の[docs/game-overview.md](../../../docs/game-overview.md)を参照。ここでは「GPU移植版としてどう使うか」に絞る。

## 使い方

```python
import jax
from kaggriculture.simulator.reset import reset
from kaggriculture.simulator.step import step_batch
from kaggriculture.simulator.action import Action
from kaggriculture.simulator import constants as C

key = jax.random.PRNGKey(0)
state = reset(key, batch_size=4096)          # [B, 2, ...] 形のState

jitted_step = jax.jit(step_batch, static_argnames=["board_size"])

# actionは文字列ではなく、constants.pyのop番号を使った整数配列で組み立てる
action = Action(
    farmer_op=jax.numpy.full((4096, 2), C.FARMER_OP_PASS, dtype=jax.numpy.int32),
    ...  # 他のフィールドも同様
)

state, reward, done = jitted_step(state, action)
```

- `reset(key, batch_size, board_size=10, starting_money=3000.0) -> State`: B局分の初期状態を作る
- `step(state, action, **config) -> (state, reward, done)`: **バッチ軸を持たない**1局分の状態遷移。`state`/`action`は`[2, ...]`(プレイヤー軸のみ)
- `step_batch(state, action, **config) -> (state, reward, done)`: `step()`をvmapしたバッチ版。`state`/`action`は`[B, 2, ...]`(`reset()`の戻り値と同じ形)。通常はこちらを使う
- `step_batch_lockstep(...)`: `step_batch`の最適化版。バッチ内の全局が同じstepを共有している(通常の学習ループの前提)場合に、日末処理のコストを削れる。前提が崩れると結果が壊れるので、詳細は[step.py](step.py)のdocstringを読んでから使うこと

`jax.jit`で包む場合、`board_size`は`static_argnames`に含める必要がある(配列のshapeを決めるため)。

## 動作仕様

- **公式シミュレータとの一致**: `scripts/generate_golden_traces.py`でKaggle提供版(CPU)から生成したトレースと、`scripts/validate_against_golden_trace.py`で全ターン突き合わせている。乱数(雑草の自然発生・店の抽選)が絡まない設定(`weedSpawnChance: 0`、`townShopUnlockInterval`を十分大きくする)なら、全フィールド・reward・doneが完全一致することを確認済み(引数無しで`validate_against_golden_trace.py`を実行すると、コミット済みの3トレースを全ターン検証する)
- **乱数**: JAXの乱数(threefry)を使うため、Pythonの`random.Random`(メルセンヌ・ツイスタ)とはビット単位で一致しない。雑草の自然発生・店の抽選は「確率分布として正しい」ことのみ保証する(具体的にどのマスに雑草が生えるかは元の実行結果と一致しなくてよい)
- **観測のマスキング**: `State`は内部シミュレータとして両プレイヤーの全情報を平等に保持している。Kaggle提供版は各プレイヤーに相手の`private`(納屋・種・持ち物)を隠した観測を渡すため、方策の入力には`State`をそのまま使わず、[observation.py](observation.py)の`build_observation(state, player)`(1プレイヤー分)/`build_observations(state)`(両プレイヤー分)を使うこと。**`State`全体をそのまま方策の入力にすると、学習中は相手の情報が丸見えになる**
- **`MAX_HANDS`/`MAX_MARKET_ORDERS`**: 元のルールに存在しない、GPU移植版固有の上限([constants.py](constants.py)参照)。配列のshapeを固定するために導入した
- **未対応の設定**: `marketParams`(品目ごとの価格曲線の上書き)は未対応。市場価格パラメータは[game_params.py](game_params.py)の固定値のみ

### `step()`の設定引数とkaggriculture.jsonのconfiguration対応

| `step()`の引数 | configurationのキー | デフォルト |
|---|---|---|
| `board_size` | `boardSize` | 10 |
| `turns_per_day` | `turnsPerDay` | 24 |
| `shed_capacity` | `shedCapacity` | 100 |
| `weed_chance` | `weedSpawnChance` | 0.005 |
| `shop_unlock_interval` | `townShopUnlockInterval` | 3 |
| `shop_sell_interval` | `townShopSellInterval` | 4 |
| `center_sell_interval` | `townCenterSellInterval` | 24 |
| `hire_mult` | `farmHandCostMult` | 1 |
| `max_shop_instances` | (無し。元コードの固定値`MAX_SHOP_INSTANCES`) | 8 |
| `episode_steps` | `episodeSteps` | 720 |

`starting_money`(`reset()`側の引数)は`startingMoney`に対応。

## 各ファイルの役割

### 中核データ構造

| ファイル | 役割 |
|---|---|
| [state.py](state.py) | ゲーム状態`State`(NamedTuple)の定義。全フィールド固定shape |
| [action.py](action.py) | 1ターン分の行動`Action`(NamedTuple)の定義 |
| [reset.py](reset.py) | 初期状態を作る |
| [observation.py](observation.py) | `State`から各プレイヤー向けのマスク済み観測を作る |
| [constants.py](constants.py) | 定数(作物/動物/タイル種類/op番号などのインデックス割り当て) |
| [game_params.py](game_params.py) | ゲームバランスの数値パラメータ(種代・収穫日数・市場基準価格など) |
| [board.py](board.py) | 盤面座標の共通ヘルパー(区画・納屋隣接マス) |

### 1ターンの処理(呼び出し順)

| ファイル | 役割 |
|---|---|
| [step.py](step.py) | 1ターン分の状態遷移を統括する`step`/`step_batch`/`step_batch_lockstep` |
| [player_turn.py](player_turn.py) | 1プレイヤー分、farmer→hands の順にユニット行動を適用 |
| [unit_actions.py](unit_actions.py) | 1ユニット分の行動の統合ディスパッチ |
| [crop_actions.py](crop_actions.py) | 作物関連の行動(PLANT/WATER/HARVEST/FERTILIZE/DIG) |
| [animal_actions.py](animal_actions.py) | 動物関連の行動(BUILD_\*/PLACE/FEED/CARE/COLLECT_FERTILIZER) |
| [inventory_actions.py](inventory_actions.py) | 持ち物・納屋関連の行動(PICKUP/DROP) |
| [market_queue.py](market_queue.py) | 市場注文キューを両プレイヤー分、スロット順に処理 |
| [market_orders.py](market_orders.py) | 閉じた式で計算できる市場注文(HIRE/BUY_LAND/BUY_SEED/BUY_ANIMAL) |
| [market_lockstep.py](market_lockstep.py) | ロックステップが必要な市場注文(SELL/BUY_PRODUCT) |
| [market.py](market.py) | 市場価格の計算式 |
| [town.py](town.py) | 街(店舗・タウンセンター)による市場消費 |
| [decay.py](decay.py) | 植物の寿命超過後の減衰 |
| [daily_refresh.py](daily_refresh.py) | 作物・動物の日次更新(日末のみ) |
| [random_events.py](random_events.py) | 雑草の自然発生・店の抽選(日末のみ) |
| [end_of_day.py](end_of_day.py) | 持ち物の払い出し・位置リセット(日末のみ) |

## 検証

```bash
# 全deterministicトレースを全ターン検証(CI/回帰確認用のデフォルト動作)
PYTHONPATH=src python3 scripts/validate_against_golden_trace.py

# 個別トレースを指定
PYTHONPATH=src python3 scripts/validate_against_golden_trace.py <trace_name> [max_steps]

# 新しいゴールデントレースを生成し直す
PYTHONPATH=src python3 scripts/generate_golden_traces.py
```
