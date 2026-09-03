"""GPU上でバッチ実行するためのゲーム状態(State)の定義。

元のkaggriculture.pyは1局分の状態を、キーが動的に増減する辞書の集まりとして
表現していた(tiles[y][x]が None/"LOCKED"/dict、shedやseedsが疎な辞書、
handsが可変長リスト、など)。

GPU上でjit/vmapするには、状態の"形(shape)"があらかじめ固定されている必要がある。
そのためこのモジュールでは、全てのフィールドを固定shapeの配列にする:

- 辞書(shed/seeds/inventory) → constants.pyで決めた並び順の固定長ベクトル
- 可変長リスト(hands)        → MAX_HANDS個で固定し、有効フラグ(*_active)で
                               実際に使っている分だけを示す
- タグ付きユニオン(tiles)    → 複数の同じshapeの配列(チャンネル)の束
- 出現店リスト(unlocked_shops)→ 店の種類ごとの出現数を表す固定長ベクトル

shapeは2通りある:

    step():       [2, board, board, ...]     (1局分、バッチ軸なし)
    step_batch(): [B, 2, board, board, ...]  (B局分。reset()もこの形で返す)

以下の各フィールドのshapeコメントは前者(step()の契約)で統一して書く。
step_batch()/reset()を使う場合は、先頭にさらにバッチ軸Bが付くと読み替える。
プレイヤー軸(2)が無いのはmarket/town/stepなど両プレイヤー共有のフィールドのみ。

NamedTupleを使っているのは、JAXがNamedTupleを自動的にpytree(jit/vmapが
中身を再帰的に辿れるデータ構造)として扱ってくれるため。
"""

from typing import NamedTuple

import jax.numpy as jnp


class State(NamedTuple):
    # ============================================================
    # 盤面(tiles) — 全て shape [2, board, board]
    # ============================================================
    # マスの種類(constants.TILE_*)
    tiles_kind: jnp.ndarray
    # PLANTなら作物インデックス(constants.CROPS)、COOP/PASTUREに動物がいるなら
    # 動物インデックス(constants.ANIMALS)。動物がまだ入っていない空のCOOP/PASTURE
    # では -1(元コードの "animal" キーが無い状態に対応する番兵値)。
    # それ以外のkind(EMPTY/LOCKED/WEED)では値に意味は無い
    tiles_crop_or_animal: jnp.ndarray
    # PLANTならplanted_day、動物がいるCOOP/PASTUREならplaced_day
    tiles_planted_or_placed_day: jnp.ndarray
    # その日すでに水やり/餌やり済みか
    tiles_watered_or_fed_today: jnp.ndarray
    # 水やり/餌やりをサボった連続日数(2でWEED化/逃亡)
    tiles_consecutive_unwatered_or_unfed: jnp.ndarray
    # 収穫可能な蓄積量
    tiles_yield_units: jnp.ndarray
    # PLANT限定: この値以降、隔ターンでyield_unitsが減っていく(-1 = 該当なし)
    tiles_max_lifespan_step: jnp.ndarray
    # PLANT限定: この日までFERTILIZEのボーナスが有効(-1 = 該当なし)
    tiles_fertilized_until_day: jnp.ndarray
    # 動物限定: その日すでにCAREしたか
    tiles_cared_today: jnp.ndarray
    # 動物限定: 未回収の肥料が1個あるか
    tiles_fertilizer_available: jnp.ndarray
    # 動物限定: 次回生産時に上乗せされるCAREボーナスの蓄積値
    tiles_pending_care_bonus: jnp.ndarray

    # ============================================================
    # ユニット位置 — shape [2, 2](末尾はx,y)、hands系は [2, MAX_HANDS, 2]
    # ============================================================
    farmer_pos: jnp.ndarray
    hands_pos: jnp.ndarray
    # そのhandスロットが今日実際に雇われているか(shape [2, MAX_HANDS])
    hands_active: jnp.ndarray
    # 今日すでに何人雇ったか(次のHIREのコスト計算に使う)。shape [2]
    hires_today: jnp.ndarray

    # ============================================================
    # 農場のその他の状態
    # ============================================================
    money: jnp.ndarray  # shape [2]
    # 解放済み区画(constants.QUADRANTS順のbool)。shape [2, N_QUADRANTS]
    unlocked_quadrants: jnp.ndarray

    # ============================================================
    # 非公開状態(private) — 本来は相手から見えないが、シミュレータ内部では
    # 普通に両方持っておき、「エージェントへの観測を作る」層で隠す
    # ============================================================
    # 納屋の在庫。shape [2, N_SHED_ITEMS]
    shed: jnp.ndarray
    # 未使用の種の数。shape [2, N_CROPS]
    seeds: jnp.ndarray
    # 主農夫が今日集めた持ち物。shape [2, N_SHED_ITEMS]
    farmer_inventory: jnp.ndarray
    # 各handが今日集めた持ち物。shape [2, MAX_HANDS, N_SHED_ITEMS]
    hands_inventory: jnp.ndarray

    # ============================================================
    # 市場(両プレイヤー共有、プレイヤー軸なし)
    # ============================================================
    # 品目ごとの在庫量。shape [N_PRODUCTS]
    # 価格はこの在庫量から都度計算する(Stateにキャッシュしない。game_market.py参照予定)
    market_inventory: jnp.ndarray

    # ============================================================
    # 街(両プレイヤー共有)
    # ============================================================
    # 出現済みの店を「種類ごとの出現数」で表す。shape [N_SHOPS]
    # (元コードのunlocked_shopsという名前のリストを、消費計算に必要な
    # 「個数」の情報だけに落とし込んだもの。出現順は消費量に影響しないため不要)
    town_shop_counts: jnp.ndarray

    # ============================================================
    # 時間・乱数(両プレイヤー共有)
    # ============================================================
    # エピソード内の通しターン数。dayやhourはここから導出する(turnsPerDay次第)
    step: jnp.ndarray  # shape []
    # エピソード開始時に決まるベースシード(jax.random.PRNGKeyの形。shape [2])。
    # 元コードの_end_of_dayは毎日 (seed * 1_000_003) ^ day から乱数器を作り直しており、
    # 「前日までの消費状況」を引き継ぐ必要が無い(ベースシードと日付だけで決まる)。
    # これに倣い、rng_keyはステップごとに更新される状態ではなく、reset時に一度
    # 決めたら変わらない値として持つ。その日の乱数が要る時は、この値と現在の
    # dayから毎回 jax.random.fold_in 等で導出する(=どの日の乱数も、前の日の
    # 計算結果を待たずに独立に計算できる)。
    rng_key: jnp.ndarray
