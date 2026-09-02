"""ゲームバランスの数値パラメータ(種代・収穫日数・市場基準価格・土地代など)。

constants.py で決めた並び順(CROPS/ANIMALS/PRODUCTS/SHOPSのインデックス)に
揃えたタプルとして定義する。dictではなく並び順の揃ったタプルにしておくことで、
そのまま `jnp.array(...)` してテンソル演算に使える。

元データの出典: data/kaggle_environments_src/kaggriculture.py
"""

from .constants import (
    ANIMALS,
    CROPS,
    FUNC_HINGE,
    FUNC_LINEAR,
    FUNC_LOG,
    FUNC_SQ,
    FUNC_SQRT,
    N_ANIMALS,
    N_CROPS,
    N_PRODUCTS,
    N_SHOPS,
    PRODUCTS,
    SHOPS,
    TILE_COOP,
    TILE_PASTURE,
)

assert len(CROPS) == N_CROPS
assert len(ANIMALS) == N_ANIMALS

# ============================================================
# 作物(CROPSと同じ並び順: WHEAT, CARROT, TOMATO, STRAWBERRY, MELON)
# ============================================================

# 種のコスト($)
CROP_SEED_COST = (10, 20, 50, 100, 80)

# 植えてから最初に収穫できるようになるまでの日数
CROP_FIRST_YIELD_DAY = (2, 2, 8, 10, 10)

# 一発収穫型(ongoing=False)は収穫量が伸びなくなる日数、
# 継続収穫型(ongoing=True)は「最初の収穫」の日数(以降はCROP_INTERVAL間隔で継続)
CROP_MAX_YIELD_DAY = (4, 3, 8, 10, 12)

# 継続収穫型の収穫間隔(日数)。一発収穫型は0(使われない)
CROP_INTERVAL = (0, 0, 1, 2, 0)

# 収穫できる最大量(継続収穫型は生涯の累計生産回数の上限でもある)
CROP_MAX_YIELD = (6, 4, 4, 4, 6)

# True = 継続収穫型(トマト・イチゴ)、False = 一発収穫型
CROP_IS_ONGOING = (False, False, True, True, False)

# ============================================================
# 動物(ANIMALSと同じ並び順: GOOSE, COW, SHEEP)
# ============================================================

# 購入コスト($)
ANIMAL_COST = (300, 400, 500)

# 配置に必要な小屋の種類(tiles配列のkindチャンネルの値。TILE_COOP/TILE_PASTURE)
ANIMAL_STRUCTURE = (TILE_COOP, TILE_PASTURE, TILE_PASTURE)

# 配置してから最初に収穫できるようになるまでの日数
ANIMAL_FIRST_YIELD_DAY = (4, 8, 6)

# 収穫の間隔(日数)。動物は上限に達しない限り無期限に生産し続ける
ANIMAL_INTERVAL = (1, 2, 3)

# 収穫せず放置できる上限(生涯の累計生産量の上限ではない。詳細はREADME参照)
ANIMAL_MAX_HELD = (4, 6, 6)

# 生産物がPRODUCTSの何番目か(GOOSE→EGG, COW→MILK, SHEEP→WOOL)。
# 動物名と生産物名が一致しない非対称性はconstants.pyのPRODUCTS定義コメント参照。
ANIMAL_PRODUCT_IDX = (PRODUCTS.index("EGG"), PRODUCTS.index("MILK"), PRODUCTS.index("WOOL"))

# ============================================================
# 土地(区画)
# ============================================================

# NE, SW, SE の順に買い足す時の価格($)。NWは常に解放済みなのでここには含まれない
# (constants.pyのQUADRANTS=("NW","NE","SW","SE")の1番目以降に対応)。
LAND_PRICES = (1000, 2000, 4000)

# ============================================================
# farm hand の雇用コスト
# ============================================================

# n番目(0-indexed)の雇用コスト = FARM_HAND_COST_MULT * fib(n)
# fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5, ...
FARM_HAND_COST_MULT = 1

# ============================================================
# 市場(PRODUCTSと同じ並び順)
# ============================================================

# 全品目共通の基準在庫量。市場はこの値からスタートし、価格はこれとの差で決まる
MARKET_I0 = 10000

# 価格の下限($)。これより安くはならない
PRICE_FLOOR = 1

# hinge関数(在庫がTを超えると急騰/急落する形状)の急峻さを決める係数
HINGE_GAIN = 8.0

# 基準在庫量I0の時の価格($)
MARKET_BASE_PRICE = (25, 35, 60, 120, 250, 50, 160, 200, 100)

# 「品薄/供給過多とみなす」規模の目安(生産能力ベースの値。詳細はREADME参照)
MARKET_T = (400, 450, 200, 100, 300, 332, 122, 105, 200)

# 在庫がI0を下回った(品薄)時に使う形状関数
MARKET_BELOW_FUNC = (
    FUNC_SQRT,   # WHEAT
    FUNC_HINGE,  # CARROT
    FUNC_HINGE,  # TOMATO
    FUNC_SQRT,   # STRAWBERRY
    FUNC_LOG,    # MELON
    FUNC_HINGE,  # EGG
    FUNC_SQRT,   # MILK
    FUNC_LOG,    # WOOL
    FUNC_LINEAR, # FERTILIZER
)

# 品薄側の価格の効き具合(T単位分だけ品薄になった時、基準価格の何倍動くか)
MARKET_BELOW_TARGET = (0.80, 1.00, 0.40, 0.70, 0.20, 0.40, 0.60, 0.20, 0.40)

# 在庫がI0を上回った(供給過多)時に使う形状関数
MARKET_ABOVE_FUNC = (
    FUNC_LOG,    # WHEAT
    FUNC_SQRT,   # CARROT
    FUNC_SQRT,   # TOMATO
    FUNC_LINEAR, # STRAWBERRY
    FUNC_SQ,     # MELON
    FUNC_LOG,    # EGG
    FUNC_LINEAR, # MILK
    FUNC_SQ,     # WOOL
    FUNC_LINEAR, # FERTILIZER
)

# 供給過多側の価格の効き具合
MARKET_ABOVE_TARGET = (0.20, 0.70, 0.60, 1.60, 3.60, 0.20, 1.60, 3.20, 0.40)

# ============================================================
# 街の店(SHOPSと同じ並び順)ごとの品目別消費量
# ============================================================
# SHOP_DEMAND[shop_idx][product_idx] = そのショップが1ティックごとに買い取る量
# (0 = 扱っていない品目)。元コードは「扱う品目数が1個だけの店は2倍量」という
# ルールだったので、ここでは最初からその倍率込みの値にしてある。
_SHOP_PRODUCTS = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}


def _build_shop_demand():
    table = [[0] * N_PRODUCTS for _ in range(N_SHOPS)]
    for shop_idx, shop in enumerate(SHOPS):
        products = _SHOP_PRODUCTS[shop]
        multiplier = 2 if len(products) == 1 else 1
        for product in products:
            table[shop_idx][PRODUCTS.index(product)] = multiplier
    return tuple(tuple(row) for row in table)


SHOP_DEMAND = _build_shop_demand()

# 街の中心(town center)が毎ティック買い取る品目。全品目 - 肥料
TOWN_CENTER_PRODUCTS = tuple(p != "FERTILIZER" for p in PRODUCTS)

# ショップの抽選は最大何インスタンスで打ち止めになるか(重複可、種類数の上限ではない)
MAX_SHOP_INSTANCES = 8
