"""episode_history.compute_turn_deltas/update_countersの契約テスト。実データに
依存せず、合成observation(conftest.fresh_obs)だけで完結する。
"""

import copy

import torch

from kaggriculture.policy.torch import distribution as D
from kaggriculture.policy.torch import history as EH


def _env_args(obs, turns_per_day=24, shed_capacity=100, hire_mult=1, max_market_orders=10):
    env = D._envs_from_observations(
        [obs], turns_per_day, shed_capacity, hire_mult, max_market_orders
    )[0]
    return env.farm, env.shed, env.seeds, env.market, env.inventories, env.day


def test_compute_turn_deltas_does_not_mutate_inventories(fresh_obs):
    obs = fresh_obs
    obs["private"]["inventories"][0]["WHEAT"] = 1
    before = copy.deepcopy(obs["private"]["inventories"])
    action = {"farmer": ["PLACE", "WHEAT"], "hands": [], "market": []}

    EH.compute_turn_deltas(
        obs["farms"][0],
        obs["private"]["shed"],
        obs["private"]["seeds"],
        obs["market"],
        obs["private"]["inventories"],
        obs["day"],
        action,
    )

    assert obs["private"]["inventories"] == before


def test_no_op_action_yields_no_deltas(fresh_obs):
    """STOP以外何もしないactionなら、生産・売買のdeltaは一切発生しない。"""
    farm, shed, seeds, market, inventories, day = _env_args(fresh_obs)
    action = {"farmer": ["EAST"], "hands": [], "market": []}
    deltas = EH.compute_turn_deltas(farm, shed, seeds, market, inventories, day, action)
    assert deltas == {}


def test_buy_seed_is_not_counted_as_bought_product(fresh_obs):
    """estimated_bought_productカテゴリはBUY_PRODUCT専用
    (BUY_SEED/BUY_ANIMAL等は対象外)。"""
    farm, shed, seeds, market, inventories, day = _env_args(fresh_obs)
    action = {"farmer": ["EAST"], "hands": [], "market": [["BUY_SEED", "WHEAT", 3]]}
    deltas = EH.compute_turn_deltas(farm, shed, seeds, market, inventories, day, action)
    assert deltas == {}


def test_buy_product_and_sell_produce_matching_deltas(fresh_obs):
    farm, shed, seeds, market, inventories, day = _env_args(fresh_obs)
    action = {"farmer": ["EAST"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 10]]}
    deltas = EH.compute_turn_deltas(farm, shed, seeds, market, inventories, day, action)
    assert deltas["estimated_bought_product"]["WHEAT"] == 10
    assert "sold" not in deltas
    assert "estimated_revenue" not in deltas

    counters = EH.update_counters({}, deltas)
    assert counters["estimated_bought_product"]["WHEAT"] == 10
    assert counters["has_ever_sold"] == {}

    # 続けてSELLすると、sold/estimated_revenueが積算される。
    farm2, shed2, seeds2, market2, inventories2, day2 = _env_args(fresh_obs)
    shed2["WHEAT"] = 10  # 直前のBUY_PRODUCTで手に入った分を反映
    sell_action = {"farmer": ["EAST"], "hands": [], "market": [["SELL", "WHEAT", 5]]}
    deltas2 = EH.compute_turn_deltas(farm2, shed2, seeds2, market2, inventories2, day2, sell_action)
    assert deltas2["sold"]["WHEAT"] == 5
    assert deltas2["estimated_revenue"]["WHEAT"] > 0

    counters2 = EH.update_counters(counters, deltas2)
    assert counters2["estimated_bought_product"]["WHEAT"] == 10
    assert counters2["sold"]["WHEAT"] == 5
    assert counters2["has_ever_sold"]["WHEAT"] is True
    # 元のcountersは書き換えない(update_countersは新しい辞書を返す)。
    assert counters["sold"] == {}


def test_counters_flow_through_act_and_evaluate(net_with_history, fresh_obs):
    """countersを渡してもact/evaluate_actionsの整合性(同じ入力→同じ出力)は保たれる。"""
    net = net_with_history
    counters = {"produced": {"WHEAT": 20}, "sold": {"WHEAT": 5}, "has_ever_sold": {"WHEAT": True}}
    action, value, log_prob, entropy, nd = D.act(net, fresh_obs, counters=counters)
    value2, log_prob2, entropy2, nd2 = D.evaluate_actions(net, fresh_obs, action, counters=counters)
    assert nd == nd2
    assert abs(value.item() - value2.item()) < 1e-4
    assert abs(log_prob.item() - log_prob2.item()) < 1e-4
    assert abs(entropy.item() - entropy2.item()) < 1e-4


def test_counters_actually_change_encoder_output(net_with_history, fresh_obs):
    """countersを渡すのと空のcountersを渡すのとで、Encoder出力(≒価値)が変わる
    (=単に無視されて死んだ引数になっていないことの確認)。"""
    net = net_with_history
    torch.manual_seed(0)
    _, value_without, *_ = D.act(net, fresh_obs, counters={})
    counters = {"estimated_revenue": {"WHEAT": 50_000}}
    torch.manual_seed(0)
    _, value_with, *_ = D.act(net, fresh_obs, counters=counters)
    assert abs(value_without.item() - value_with.item()) > 1e-9


def test_use_episode_history_none_is_rejected(net_with_history, fresh_obs):
    """use_episode_history=Trueのネットにcounters=None(省略)で呼ぶと、
    「初期状態(全部ゼロ)」のつもりが実は渡し忘れ、という事故を防ぐため
    ValueErrorになる。"""
    try:
        D.act(net_with_history, fresh_obs)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_counters_rejected_when_not_using_episode_history(net, fresh_obs):
    """use_episode_history=False(既定)のネットにcountersを渡すとエラーになる
    (黙って無視されない)。"""
    try:
        D.act(net, fresh_obs, counters={"produced": {"WHEAT": 1}})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
