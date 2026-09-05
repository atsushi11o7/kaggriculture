"""方策ネットが期待する疎特徴量(SparseVector)の語彙定義。

方策ネットはnn.EmbeddingBag(mode="sum")で疎特徴量を埋め込む。SparseVectorは
(index, value)のペアの集まりで、EmbeddingBagは sum(embedding[index[i]] * value[i])
を計算する。カテゴリ特徴はvalue=1.0、連続値(個数・日数等)はvalue=正規化した値、
という使い分けにする。

このモジュールは「ネットワークがどの添字をどう解釈するか」という契約だけを持つ。
生の観測(リプレイJSON、自前シミュレータのState等)からこの語彙のSparseVectorへ
変換する処理は、データソースごとに別パッケージが担う。

盤面は「自分の100マス+相手の100マス+その他のゾーン(所持金・納屋・種・持ち物・
市場・街・ターン)」をそれぞれ1トークンとしてエンコードする想定。各トークンの
owner(誰の情報か)とzone(何の種類の情報か)は局面によらず固定なので、
EmbeddingBagの外側で別途足し込む(ネットワーク本体側の実装で対応させる)。
"""

from dataclasses import dataclass, field

from kaggriculture.simulator import constants as C

_next_index = 0


def _alloc(n: int) -> range:
    """語彙にn個のスロットを新規に割り当て、その添字範囲を返す。"""
    global _next_index
    r = range(_next_index, _next_index + n)
    _next_index += n
    return r


# --- タイル特徴 ---
TILE_KIND = _alloc(6)  # constants.TILE_EMPTY..TILE_PASTUREと同じ並び
TILE_CROP = _alloc(C.N_CROPS)  # PLANT限定: 作物
TILE_ANIMAL = _alloc(C.N_ANIMALS)  # COOP/PASTURE限定: 動物
TILE_CARE_DONE_TODAY = _alloc(1)  # 水やり/給餌済みか(PLANT/動物共通の1スロット)
TILE_CARED_TODAY = _alloc(1)  # 動物限定: 世話済みか
TILE_FERTILIZED_ACTIVE = _alloc(1)  # PLANT限定: 施肥ボーナス有効中か
TILE_FERTILIZER_AVAILABLE = _alloc(1)  # 動物限定: 未回収の肥料があるか
TILE_IS_FARMER = _alloc(1)  # farmerがこのマスにいるか
TILE_HAND_COUNT = _alloc(1)  # このマスにいるhandの数(value=個数)
TILE_AGE = _alloc(1)  # 植付/配置からの経過日数(value=正規化値)
TILE_YIELD_UNITS = _alloc(1)  # 収穫可能量(value=正規化値)
TILE_CONSECUTIVE_UNCARED = _alloc(1)  # 未水やり/未給餌の連続日数(value=正規化値)
TILE_PENDING_CARE_BONUS = _alloc(1)  # 動物限定: 蓄積CAREボーナス(value=正規化値)
TILE_LIFESPAN_REMAINING = _alloc(1)  # PLANT限定: 一発収穫型のみ。max_lifespan_step
# (減衰が始まる通しターン数)から現在の通しターン数を引いた残りターン数(value=
# 正規化値)。継続収穫型(max_lifespan_step=-1)には無い。「そろそろ枯れる」を
# 明示的に伝える(yield_units等から間接的に読み取らせるより素直)。

# --- プレイヤー情報(自分・相手共通の1トークン) ---
PLAYER_MONEY = _alloc(1)  # value=正規化した所持金
PLAYER_UNLOCKED_QUADRANT = _alloc(C.N_QUADRANTS)  # 解放済み区画ごとに1スロット
PLAYER_HIRES_TODAY = _alloc(1)  # value=正規化した今日の雇用人数

# --- 納屋(自分のみ実数値、相手は常にゼロ=空のトークン) ---
SHED_ITEM = _alloc(C.N_SHED_ITEMS)  # value=品目ごとの個数(正規化)

# --- 残り種(自分のみ) ---
SEED_CROP = _alloc(C.N_CROPS)  # value=作物ごとの残り種数(正規化)

# --- 持ち物(自分のみ。farmerとhands合算の1トークン) ---
INVENTORY_ITEM = _alloc(C.N_SHED_ITEMS)  # value=品目ごとの個数(正規化)

# --- 市場(共有) ---
MARKET_PRODUCT = _alloc(C.N_PRODUCTS)  # value=在庫量(正規化)
MARKET_PRICE = _alloc(C.N_PRODUCTS)  # value=現在価格(正規化)。在庫から計算できる値だが、
# 品目ごとに違う非線形カーブ(price関数)をネットワークに逆算させずに済むよう、
# 価格そのものも直接与える。

# --- 街(共有) ---
TOWN_SHOP = _alloc(C.N_SHOPS)  # value=出現数

# --- ターン(共有) ---
TURN_DAY = _alloc(1)  # value=正規化した経過日数
TURN_HOUR = _alloc(1)  # value=正規化した日内ターン

# ============================================================
# 行動候補(デコーダ側)の語彙
# ============================================================
# 「今合法な行動候補だけ」を列挙してデコーダに渡す(全行動空間に対する固定サイズの
# 出力ではない)。1候補は(op, item)の組で表し、opはfarmer/hand用とmarket用で
# 別々の埋め込みを持つ。itemは対象(作物・動物・品目)ごとに、盤面側で使っている
# 埋め込み(TILE_CROP/TILE_ANIMAL/SHED_ITEM/MARKET_PRODUCT)をそのまま再利用する
# (例: 「MELONを植える」候補と「タイルにMELONが植わっている」事実は同じ埋め込みを
# 共有した方が、観測と行動の対応関係を学習しやすい)。
ACTION_FARMER_OP = _alloc(C.N_FARMER_OPS)  # constants.FARMER_OP_NAMESと同じ並び
ACTION_MARKET_OP = _alloc(C.N_MARKET_OPS)  # constants.MARKET_OP_NAMESと同じ並び

VOCAB_SIZE = _next_index


@dataclass
class SparseVector:
    """1トークン分の疎特徴量。EmbeddingBagにそのまま渡せる(index, value)の列。"""

    index: list[int] = field(default_factory=list)
    value: list[float] = field(default_factory=list)

    def add(self, index: int, value: float = 1.0) -> None:
        self.index.append(index)
        self.value.append(value)
