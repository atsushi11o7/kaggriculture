"""kaggle-environments版kaggriculture.pyの定数テーブルを、テンソルのインデックスとして
使える形(名前の並び順が固定されたタプル)に変換したもの。

辞書のキー(例: "WHEAT")ではテンソルの軸に使えないので、ここで各カテゴリの
「並び順」を1箇所に固定し、以降のコードは全てこの並び順(インデックス)で
配列にアクセスする。並び順自体に意味はなく、常に一貫していることだけが重要。

元データの出典: data/kaggle_environments_src/kaggriculture.py
"""

# --- 作物 ---
# 元のCROPS辞書のキー順(Python 3.7+の辞書は挿入順を保持するので、これが元の並び順と一致する)
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
N_CROPS = len(CROPS)

# --- 動物 ---
ANIMALS = ("GOOSE", "COW", "SHEEP")
N_ANIMALS = len(ANIMALS)

# --- 市場で取引される全品目(作物の収穫物 + 動物の生産物 + 肥料) ---
# 元のPRODUCTSリストとそのまま同じ並び
#
# 注意: 作物と動物で「CROPS/ANIMALSのインデックス」と「PRODUCTSのインデックス」の
# 対応関係が非対称。作物は植える対象と収穫物が同じ名前(例: "WHEAT"のcrop_idxと
# product_idxは名前が一致する)。一方、動物は飼う対象と生産物の名前が別
# (GOOSE→EGG, COW→MILK, SHEEP→WOOL)なので、animal_idxからproduct_idxへの変換には
# 対応表が必要になる。
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
N_PRODUCTS = len(PRODUCTS)

# --- 納屋(shed)に入りうる品目全体 = 収穫物・肥料 + 動物そのもの ---
# 動物はPLACEするまでの間、家畜そのものが納屋の在庫として存在しうる
SHED_ITEMS = PRODUCTS + ANIMALS
N_SHED_ITEMS = len(SHED_ITEMS)

# --- 街の店 ---
# townShopUnlockInterval(デフォルト3日)ごとに、この8種類から重複ありでランダムに
# 1つ抽選されて出現する(最大8軒まで)。出現した店はゲーム終了まで居座り、
# townShopSellInterval(デフォルト4ターン)ごとに以下の品目を1つずつ市場から
# 買い取り続ける(単品目の店は2倍量: YARN_STORE, PET_CAFE)。
#   BAKERY:         卵, 小麦
#   PIZZA_SHOP:     牛乳, トマト, 小麦
#   BRUNCH_SPOT:    卵, 小麦, イチゴ
#   YARN_STORE:     羊毛(2倍)
#   ICE_CREAM_SHOP: イチゴ, 牛乳, 小麦
#   PET_CAFE:       人参(2倍)
#   SMOOTHIE_SHOP:  イチゴ, 牛乳
#   FARMERS_MARKET: 小麦, 人参, トマト, イチゴ
SHOPS = (
    "BAKERY",
    "PIZZA_SHOP",
    "BRUNCH_SPOT",
    "YARN_STORE",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "SMOOTHIE_SHOP",
    "FARMERS_MARKET",
)
N_SHOPS = len(SHOPS)

# --- 区画 ---
# 各プレイヤーの盤面(デフォルト10×10)は4つの5×5区画に分かれている。
# 開始時はNWだけ解放済み。残りはBUY_LANDでNE→SW→SEの順に$1,000/$2,000/$4,000で
# 買い足していく(元コードのLAND_ORDER=["NE","SW","SE"]に対応。NWは常に解放済みなので
# LAND_ORDERには含まれないが、こちらのQUADRANTSは4区画全部を含む固定の並びにしている)。
QUADRANTS = ("NW", "NE", "SW", "SE")
N_QUADRANTS = len(QUADRANTS)

# --- タイルの種類(tiles配列の"kind"チャンネルの値) ---
# 元コードでは1マスの状態が None / 文字列"LOCKED" / dict(kind=...) と型が
# 入り混じっていた。テンソルの1チャンネルに乗せるため、整数の列挙値に振り直す。
TILE_EMPTY = 0  # 元: None                          何もない解放済みの空きマス
TILE_LOCKED = 1  # 元: "LOCKED"                     未解放区画。踏めるが操作不可
TILE_PLANT = 2  # 元: {"kind": "PLANT", ...}        作物が植わっている
TILE_WEED = 3  # 元: {"kind": "WEED"}               雑草。DIGしないと使えない
TILE_COOP = 4  # 元: {"kind": "COOP", ...}          ガチョウ用の小屋(動物の有無は別チャンネル)
TILE_PASTURE = 5  # 元: {"kind": "PASTURE", ...}    牛/羊用の牧草地(同上)

# --- 市場価格カーブの形状関数(元コードの below_func / above_func の文字列) ---
# 品目ごとに在庫が基準値I0を下回った時(below)と上回った時(above)で別々の形状関数を
# 使って価格を決める(詳細はgame_params.pyのMARKET_*、および docs/simulator-io-spec.md
# または元コードの _shape 関数を参照)。文字列のままではテンソルに乗らないため、
# TILE_*と同じ考え方で整数の列挙値に振り直す。
FUNC_LINEAR = 0
FUNC_SQ = 1
FUNC_SQRT = 2
FUNC_LOG = 3
FUNC_LOG10 = 4
FUNC_HINGE = 5

# --- 農夫/handが取りうる行動(元コードのop文字列) ---
# 文字列のままではテンソルに乗らないため、他の列挙値と同じ考え方で整数に振り直す。
FARMER_OP_NORTH = 0
FARMER_OP_SOUTH = 1
FARMER_OP_EAST = 2
FARMER_OP_WEST = 3
FARMER_OP_PASS = 4
FARMER_OP_PICKUP = 5
FARMER_OP_PLANT = 6
FARMER_OP_WATER = 7
FARMER_OP_HARVEST = 8
FARMER_OP_FERTILIZE = 9
FARMER_OP_BUILD_COOP = 10
FARMER_OP_BUILD_PASTURE = 11
FARMER_OP_DIG = 12
FARMER_OP_PLACE = 13
FARMER_OP_FEED = 14
FARMER_OP_COLLECT_FERTILIZER = 15
FARMER_OP_CARE = 16
FARMER_OP_DROP = 17
N_FARMER_OPS = 18

# FARMER_OP_*と同じ並び順の文字列(元コードのop文字列)。トレースの行動文字列を
# 整数に変換する時などに FARMER_OP_NAMES.index(name) の形で使う。
FARMER_OP_NAMES = (
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PASS",
    "PICKUP",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "DIG",
    "PLACE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
    "DROP",
)

# 各opに対応する移動量(dx, dy)。移動系(NORTH/SOUTH/EAST/WEST)以外は(0, 0) =
# その場から動かない。FARMER_OP_*と同じ並び順(インデックス)。
FARMER_OP_MOVE_DELTA = (
    (0, -1),  # NORTH
    (0, 1),  # SOUTH
    (1, 0),  # EAST
    (-1, 0),  # WEST
    (0, 0),  # PASS
    (0, 0),  # PICKUP
    (0, 0),  # PLANT
    (0, 0),  # WATER
    (0, 0),  # HARVEST
    (0, 0),  # FERTILIZE
    (0, 0),  # BUILD_COOP
    (0, 0),  # BUILD_PASTURE
    (0, 0),  # DIG
    (0, 0),  # PLACE
    (0, 0),  # FEED
    (0, 0),  # COLLECT_FERTILIZER
    (0, 0),  # CARE
    (0, 0),  # DROP
)

# --- 市場注文の種類(farmer/handのopとは別の行動チャンネル) ---
MARKET_OP_BUY_SEED = 0
MARKET_OP_BUY_PRODUCT = 1
MARKET_OP_BUY_ANIMAL = 2
MARKET_OP_SELL = 3
MARKET_OP_HIRE = 4
MARKET_OP_BUY_LAND = 5
N_MARKET_OPS = 6

# MARKET_OP_*と同じ並び順の文字列。
MARKET_OP_NAMES = ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND")

# --- GPU版で新たに導入する上限(元のルールには存在しない) ---
# 1日に同時に存在しうるhandsの最大数。CPU版はPythonリストで無制限に伸びるが、
# GPU版は配列のshapeを固定する必要があるため上限を設ける。
# n人目の雇用コストはfib(n)倍率で増えるため、1日に20人雇うだけで累計$17,710と
# 現実的な資金感からは急激に非現実的なコストになる。一方この上限を上げても
# バッチ次元・品目数にしか効かないコストなので安価であり、安全マージンを
# 大きく取って32とした。実際の対戦でこの上限に迫った/達したケースが確認されたら
# 引き上げを検討する。
MAX_HANDS = 32

# 1ターンに処理される市場注文の最大数。元コードのmaxMarketOrdersPerTurn設定の
# デフォルト値(10)。この値を超える設定でepisodeを作る場合はここも合わせて
# 変更する必要がある(このシミュレータ側の新規制約ではなく、configの既定値をそのまま採用)。
MAX_MARKET_ORDERS = 10


# --- タイルの種類に対する判定 ---
# 複数モジュールから重複して書かれていたため、TILE_*の定義元であるここに集約した。
def is_plant(tile_kind):
    return tile_kind == TILE_PLANT


def is_animal_structure(tile_kind):
    """COOP/PASTUREか(動物がいるかどうかは問わない)。"""
    return (tile_kind == TILE_COOP) | (tile_kind == TILE_PASTURE)
