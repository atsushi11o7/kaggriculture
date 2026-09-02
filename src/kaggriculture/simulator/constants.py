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
SHOPS = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
         "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")
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

# --- GPU版で新たに導入する上限(元のルールには存在しない) ---
# 1日に同時に存在しうるhandsの最大数。CPU版はPythonリストで無制限に伸びるが、
# GPU版は配列のshapeを固定する必要があるため上限を設ける。
# 経済的な妥当性の検討はREADME/会話ログ参照。実際の対戦でこの上限に迫った/達した
# ケースが確認されたら引き上げを検討する。
MAX_HANDS = 32
