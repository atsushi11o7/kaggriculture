"""観測と行動候補で共有する疎特徴量の語彙。

品目識別子は観測ゾーンと行動候補で共有し、文脈はzoneや専用マーカーで表す。
単独スカラーはLayerNormで大きさを失うため、存在マーカーと量bucketを併用する。
"""

from dataclasses import dataclass, field

from kaggriculture.rules import constants as C

BOARD_SIZE = 10

# 方策上の数量上限。BUY_SEED等では理論上これを超える合法手もあり得る。
MAX_ACTION_QUANTITY = 100

_next_index = 0


def _alloc(n: int) -> range:
    """語彙にn個のスロットを新規に割り当て、その添字範囲を返す。"""
    global _next_index
    r = range(_next_index, _next_index + n)
    _next_index += n
    return r


# log正規化後も低数量を区別できる16段階。
N_MAGNITUDE_BUCKETS = 16


def bucket_index(normalized: float, n_buckets: int = N_MAGNITUDE_BUCKETS) -> int:
    """[0, 1]付近に正規化した値をバケット添字(0..n_buckets-1)に離散化する。"""
    return min(max(int(normalized * n_buckets), 0), n_buckets - 1)


# --- 実体(作物・動物・生産物)の共有識別子 ---
ENTITY_ITEM = _alloc(C.N_SHED_ITEMS)  # constants.SHED_ITEMS(=PRODUCTS+ANIMALS)と同じ並び
# ENTITY_ITEM[i]の個数/価格バケット。添字は entity_idx * N_MAGNITUDE_BUCKETS + bucket。
ENTITY_MAGNITUDE_BUCKET = _alloc(C.N_SHED_ITEMS * N_MAGNITUDE_BUCKETS)


def entity_index(name: str) -> int:
    """品目名(constants.SHED_ITEMSに含まれるもの)のENTITY_ITEM語彙添字を引く。"""
    return ENTITY_ITEM[C.SHED_ITEMS.index(name)]


# --- タイル特徴(作物・動物の識別自体はENTITY_ITEMを使う) ---
# タイルの連続値はkind/itemと共存し、LayerNorm後も方向の差が残るためbucket化しない。
TILE_KIND = _alloc(6)  # constants.TILE_EMPTY..TILE_PASTUREと同じ並び
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
PLAYER_MONEY = _alloc(1)  # 所持金があることの存在マーカー(value=1.0固定)
PLAYER_MONEY_MAGNITUDE_BUCKET = _alloc(N_MAGNITUDE_BUCKETS)  # 所持金の量バケット(離散)
PLAYER_MONEY_CONTINUOUS = _alloc(1)  # 所持金(value=正規化値、bucketの離散化損失を補う)
PLAYER_UNLOCKED_QUADRANT = _alloc(C.N_QUADRANTS)  # 解放済み区画ごとに1スロット
PLAYER_HIRES_TODAY = _alloc(1)  # value=正規化した今日の雇用人数


def _alloc_counter(n_items: int) -> tuple[range, range, range]:
    """累積値用に、品目別の存在マーカー・bucket・連続値を確保する。"""
    return _alloc(n_items), _alloc(n_items * N_MAGNITUDE_BUCKETS), _alloc(n_items)


# --- 自分の累積実績 ---
# produced/soldは確定値。ESTIMATED項目は同時市場処理を再現しない自分視点の見積もりで、
# reward shapingの確定実績には使わない。
OWN_PRODUCED, OWN_PRODUCED_MAGNITUDE_BUCKET, OWN_PRODUCED_CONTINUOUS = _alloc_counter(
    C.N_PRODUCTS
)  # HARVEST/COLLECT_FERTILIZERでinventoryへ入った累積量(decode_state参照)
OWN_SOLD, OWN_SOLD_MAGNITUDE_BUCKET, OWN_SOLD_CONTINUOUS = _alloc_counter(C.N_PRODUCTS)
(
    OWN_ESTIMATED_BOUGHT_PRODUCT,
    OWN_ESTIMATED_BOUGHT_PRODUCT_MAGNITUDE_BUCKET,
    (OWN_ESTIMATED_BOUGHT_PRODUCT_CONTINUOUS),
) = _alloc_counter(C.N_PRODUCTS)  # BUY_PRODUCTのみ(BUY_SEED/BUY_ANIMALは含まない)
(
    OWN_ESTIMATED_REVENUE,
    OWN_ESTIMATED_REVENUE_MAGNITUDE_BUCKET,
    (OWN_ESTIMATED_REVENUE_CONTINUOUS),
) = _alloc_counter(C.N_PRODUCTS)  # SELLで得た累積金額の見積もり
OWN_HAS_EVER_SOLD = _alloc(C.N_PRODUCTS)  # OWN_SOLD>0から導出可能だが、明示フラグとしても渡す

# --- 街(共有) ---
TOWN_SHOP = _alloc(C.N_SHOPS)  # 出現マーカー(value=1.0固定)
TOWN_SHOP_MAGNITUDE_BUCKET = _alloc(C.N_SHOPS * N_MAGNITUDE_BUCKETS)  # 出現数の量バケット

# --- ターン(共有) ---
TURN_DAY = _alloc(1)  # value=正規化した経過日数
TURN_HOUR = _alloc(1)  # value=正規化した日内ターン

# ============================================================
# 行動候補(デコーダ側)の語彙
# ============================================================
# 合法な(op, item)だけを列挙する。itemは観測と同じENTITY_ITEMを共有する。
ACTION_FARMER_OP = _alloc(C.N_FARMER_OPS)  # constants.FARMER_OP_NAMESと同じ並び
ACTION_MARKET_OP = _alloc(C.N_MARKET_OPS)  # constants.MARKET_OP_NAMESと同じ並び
# 可変長の市場注文を打ち切るデコード専用候補。
ACTION_MARKET_STOP = _alloc(1)
# 市場キューを1スロット進め、後続注文の生成を続けるデコード専用候補。
ACTION_MARKET_WAIT = _alloc(1)
# 数量ごとに別のone-hot添字を使い、LayerNorm後も区別可能なcategorical分布にする。
ACTION_QUANTITY_VALUE = _alloc(MAX_ACTION_QUANTITY)  # 量k(1-indexed) → ACTION_QUANTITY_VALUE[k-1]
ACTION_QUANTITY_CONTINUOUS = _alloc(1)  # value=正規化log(量)。量ごとの順序関係を補う

# ============================================================
# デコーダ入力(ユニット固有コンテキスト)の語彙
# ============================================================
# unit位置はEncoderと共有する盤面位置埋め込みとしてmodel側で加える。
#
# ENTITY_ITEMがunitの持ち物由来であることを、直前の行動対象と区別する。
UNIT_CONTEXT_INVENTORY = _alloc(1)

VOCAB_SIZE = _next_index


@dataclass
class SparseVector:
    """1トークン分の疎特徴量。EmbeddingBagにそのまま渡せる(index, value)の列。"""

    index: list[int] = field(default_factory=list)
    value: list[float] = field(default_factory=list)

    def add(self, index: int, value: float = 1.0) -> None:
        self.index.append(index)
        self.value.append(value)

    def extend(self, other: "SparseVector") -> None:
        self.index.extend(other.index)
        self.value.extend(other.value)
