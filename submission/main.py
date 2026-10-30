"""Kaggriculture提出エージェント。

kaggle-environmentsはこのファイルと同じディレクトリをimport時だけsys.pathへ
追加するため(kaggle_environments/agent.pyのget_last_callable参照)、
`kaggriculture.*`のimportは全てこのファイルの先頭(モジュールレベル)で行う。
agent()の中で新たにimportするとsys.pathが戻された後になり失敗し得るため避ける。

raw観測日(day)が10以下ならopening専用モデル、11以上ならday12+専用モデルで推論する
(2モデル構成、それぞれ別にBC学習)。openingモデルの出力には、土地購入期限を過ぎて
いるのにBUY_LANDが含まれていない場合に強制する反応的な安全網(land_rule.py)を
重ねる。

このファイル・model_weights_opening.pt・model_weights_day12plus.pt・
kaggriculture/一式をtar.gzにまとめてルートへ置いたものを提出する
(submission/build_submission.sh参照)。
"""

import os

import torch

import kaggriculture
from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.opening import land_rule
from kaggriculture.policy.torch import history as H
from kaggriculture.policy.torch import model as M
from kaggriculture.policy.torch import policy as D
from kaggriculture.rules import constants as C

# main.py自体はkaggle-environmentsにexec()され、__file__を持たない
# (kaggle_environments/agent.py: get_last_callable参照)。代わりに、通常の
# import機構で__file__が設定されるkaggriculture パッケージから、
# 提出物のルート(main.py/model_weights_*.ptと同じ階層)を逆算する。
_SUBMISSION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(kaggriculture.__file__)))

torch.set_num_threads(1)  # actTimeout=1秒に対し、単一推論でのスレッド起動オーバーヘッドを避ける

OPENING_LAST_OBSERVATION_DAY = 10


def _load_net(filename: str) -> M.PolicyValueNet:
    checkpoint = torch.load(os.path.join(_SUBMISSION_DIR, filename), map_location="cpu")
    net = M.PolicyValueNet(
        ModelConfig(**dict(checkpoint["config"]["model"])),
        actor_only=checkpoint.get("actor_only", False),
    )
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    return net


_opening_net = _load_net("model_weights_opening.pt")
_day12plus_net = _load_net("model_weights_day12plus.pt")
_counters = {}
_previous_obs = None
_previous_action = None


def _history_input(obs):
    global _counters, _previous_obs, _previous_action
    if obs.get("step", 0) == 0 or _previous_obs is None or _previous_action is None:
        _counters = {}
    else:
        player = _previous_obs["player"]
        deltas = H.compute_turn_deltas(
            _previous_obs["farms"][player],
            _previous_obs["private"]["shed"],
            _previous_obs["private"]["seeds"],
            _previous_obs["market"],
            _previous_obs["private"]["inventories"],
            _previous_obs["day"],
            _previous_action,
        )
        _counters = H.update_counters(_counters, deltas)
    return _counters


def _ensure_land_purchase(action: dict, obs: dict) -> None:
    """土地購入期限を過ぎているのにBUY_LANDが含まれていなければ強制する。

    openingモデルはBCで学習した固定量の推論であり、実際のcash軌道が教師データと
    ズレた場合に土地購入のタイミングを逃しても再挑戦できない。land_rule.py自身の
    判定(day・cashから見て買うべきかどうか)を都度上書きで補う。
    """
    market = action.setdefault("market", [])
    if ["BUY_LAND"] in market:
        return
    if land_rule.next_forced_purchase(obs) is None:
        return
    if len(market) >= C.MAX_MARKET_ORDERS:
        market.pop()
    market.insert(0, ["BUY_LAND"])


def agent(obs):
    global _previous_obs, _previous_action
    try:
        counters = _history_input(obs)
        if obs["day"] <= OPENING_LAST_OBSERVATION_DAY:
            action = D.predict_action(_opening_net, obs, counters=counters)
        else:
            action = D.predict_action(_day12plus_net, obs, counters=counters)
        _ensure_land_purchase(action, obs)
        _previous_obs = obs
        _previous_action = action
        return action
    except Exception as e:  # noqa: BLE001 - 対局を止めないための意図的なcatch-all
        # 万一未知の観測でpredict_actionが例外を出しても、その1ターンをPASSで
        # 凌いで対局自体を継続させる(actTimeout=1秒でリトライの余地は無いため)。
        print(f"predict_action failed, falling back to PASS: {e!r}")
        return {"farmer": ["PASS"], "hands": [], "market": []}
