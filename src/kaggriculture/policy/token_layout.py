"""Encoderのトークン順とDecoderの固定スロット位置を定義する。

tokenize.pyの出力順と必ず一致させる。埋め込み処理はmodel.pyが持つ。
"""

from kaggriculture.policy import vocab as V
from kaggriculture.simulator import constants as C

N_TILE_TOKENS = 2 * V.BOARD_SIZE * V.BOARD_SIZE  # 自分の盤面 + 相手の盤面

# --- デコーダの決定スロット位置ID ---
# hand数や数量スロットの有無で意味が変わらないよう、役割ごとに2位置を予約する。
FARMER_POSITION = 0
FARMER_QUANTITY_POSITION = 1
_HAND_BASE = 2
_MARKET_BASE = _HAND_BASE + 2 * C.MAX_HANDS


def hand_position(h: int) -> int:
    """hand_h(op,item)の決定位置。"""
    return _HAND_BASE + 2 * h


def hand_quantity_position(h: int) -> int:
    """hand_hの数量(PICKUPの場合のみ)の決定位置。"""
    return _HAND_BASE + 2 * h + 1


def market_position(m: int) -> int:
    """市場注文m件目(op,item)の決定位置。"""
    return _MARKET_BASE + 2 * m


def market_quantity_position(m: int) -> int:
    """市場注文m件目の数量(数量を伴うopの場合のみ)の決定位置。"""
    return _MARKET_BASE + 2 * m + 1


MAX_DECODE_LEN = _MARKET_BASE + 2 * (C.MAX_MARKET_ORDERS + 1)  # 市場注文(STOP込み)の上限まで


# --- エンコーダのトークンごとのowner/zone/position ---
OWNER_SHARED, OWNER_OWN, OWNER_OPP = 0, 1, 2
N_OWNERS = 3

(
    ZONE_CLS,
    ZONE_TILE,
    ZONE_PLAYER_INFO,
    ZONE_SHED,
    ZONE_SEEDS,
    ZONE_INVENTORY,
    ZONE_MARKET_INVENTORY,
    ZONE_MARKET_PRICE,
    ZONE_TOWN,
    ZONE_TURN,
    ZONE_OWN_COUNTERS,
) = range(11)
N_ZONES = 11

# 盤面位置はEncoder/Decoderで共有し、位置を持たないtokenにはNO_POSITIONを使う。
NO_POSITION = N_TILE_TOKENS // 2
N_POSITIONS = NO_POSITION + 1

# 自分/相手の盤面と公開情報、自分のprivate、履歴を並べる。
# 相手のprivateは下のcritic専用token列だけに含める。
_TOKEN_OWNER_ZONE_POSITION = (
    [(OWNER_OWN, ZONE_TILE, i) for i in range(NO_POSITION)]
    + [(OWNER_OPP, ZONE_TILE, i) for i in range(NO_POSITION)]
    + [
        (OWNER_OWN, ZONE_PLAYER_INFO, NO_POSITION),
        (OWNER_OPP, ZONE_PLAYER_INFO, NO_POSITION),
        (OWNER_OWN, ZONE_SHED, NO_POSITION),
        (OWNER_OWN, ZONE_SEEDS, NO_POSITION),
        (OWNER_OWN, ZONE_INVENTORY, NO_POSITION),
        (OWNER_SHARED, ZONE_MARKET_INVENTORY, NO_POSITION),
        (OWNER_SHARED, ZONE_MARKET_PRICE, NO_POSITION),
        (OWNER_SHARED, ZONE_TOWN, NO_POSITION),
        (OWNER_SHARED, ZONE_TURN, NO_POSITION),
        (OWNER_OWN, ZONE_OWN_COUNTERS, NO_POSITION),
    ]
)
NUM_WORDS_ENCODER = len(_TOKEN_OWNER_ZONE_POSITION)  # tokenize.get_encoder_inputの出力トークン数

# CLSトークン(先頭に追加)の分を足した(owner, zone, position)の並び
TOKEN_OWNER_ZONE_POSITION_WITH_CLS = [
    (OWNER_SHARED, ZONE_CLS, NO_POSITION)
] + _TOKEN_OWNER_ZONE_POSITION


# --- 非対称critic専用の小さな非公開トークン列 ---
# 公開盤面はactor Encoderから再利用し、両者のprivateだけをunit位置つきで並べる。
PRIVILEGED_TOKENS_PER_OWNER = 3 + C.MAX_HANDS
NUM_PRIVILEGED_TOKENS = 2 * PRIVILEGED_TOKENS_PER_OWNER

_PRIVILEGED_OWNER_ZONE = []
for owner in (OWNER_OWN, OWNER_OPP):
    _PRIVILEGED_OWNER_ZONE.extend(
        [(owner, ZONE_SHED), (owner, ZONE_SEEDS), (owner, ZONE_INVENTORY)]
    )
    _PRIVILEGED_OWNER_ZONE.extend((owner, ZONE_INVENTORY) for _ in range(C.MAX_HANDS))

PRIVILEGED_OWNER_ZONE_WITH_CLS = [(OWNER_SHARED, ZONE_CLS)] + _PRIVILEGED_OWNER_ZONE
