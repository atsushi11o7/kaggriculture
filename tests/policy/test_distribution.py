"""act()/evaluate_actions()の契約テスト。実データに依存せず、合成observation
(conftest.fresh_obs)だけで完結する。

data/replays/を使った実データ回帰テストはtest_replay_regression.py参照
(data/がgitignore対象のため、存在する環境でのみ実行される)。
"""

import copy

import pytest
import torch

from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.torch import actions as A
from kaggriculture.policy.torch import distribution as D
from kaggriculture.simulator import constants as C


@pytest.mark.parametrize("max_market_orders", [0, -1, C.MAX_MARKET_ORDERS + 1])
def test_act_rejects_invalid_max_market_orders(net, fresh_obs, max_market_orders):
    """max_market_orders=0はkaggriculture.json上あり得ない値(min=1)だが、以前は
    黙って受理し「上限0のはずが1件だけ注文が通る」形で壊れていた
    (_decode_turn_genの市場注文ループは非STOP候補を選んだ*後*に上限を判定する
    ため)。設定ミスを黙って無視せず、はっきり拒否することを確認する。
    """
    with pytest.raises(ValueError):
        D.act(net, fresh_obs, max_market_orders=max_market_orders)


def test_decode_state_does_not_share_observation_inventories(fresh_obs):
    env = D._envs_from_observations([fresh_obs], 24, 100, 1, C.MAX_MARKET_ORDERS)[0]
    inventories = env.inventories

    inventories[0]["WHEAT"] = 1

    assert "WHEAT" not in fresh_obs["private"]["inventories"][0]


def test_validate_policy_action_accepts_representable_action(fresh_obs):
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    D.validate_policy_action(fresh_obs, action)  # 例外が出ないことを確認


def test_validate_policy_action_rejects_unrepresentable_action(fresh_obs):
    """farmerは空きタイルにいるのでHARVESTは候補に無い。"""
    action = {"farmer": ["HARVEST"], "hands": [], "market": []}
    with pytest.raises(ValueError):
        D.validate_policy_action(fresh_obs, action)


def test_validate_policy_action_agrees_with_evaluate_policy(net, fresh_obs):
    """事前検証(ネットワーク不使用)の合否が、実際にevaluate_policyを呼んだ場合の
    合否(ValueErrorになるかどうか)と一致することを確認する。
    """
    valid_action = {"farmer": ["PASS"], "hands": [], "market": []}
    invalid_action = {"farmer": ["HARVEST"], "hands": [], "market": []}

    D.validate_policy_action(fresh_obs, valid_action)
    D.evaluate_policy(net, fresh_obs, valid_action)  # どちらも例外を出さない

    with pytest.raises(ValueError):
        D.validate_policy_action(fresh_obs, invalid_action)
    with pytest.raises(ValueError):
        D.evaluate_policy(net, fresh_obs, invalid_action)


def test_normalize_expert_action_leaves_valid_action_unchanged(fresh_obs):
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    assert D.normalize_expert_action(fresh_obs, action) == action


def test_normalize_expert_action_maps_nonpositive_unit_quantity_to_pass(fresh_obs):
    obs = copy.deepcopy(fresh_obs)
    obs["private"]["shed"]["WHEAT"] = 1
    action = {"farmer": ["PICKUP", "WHEAT", 0], "hands": [], "market": []}

    normalized = D.normalize_expert_action(obs, action)

    assert normalized["farmer"] == ["PASS"]
    D.validate_policy_action(obs, normalized)


def test_normalize_expert_action_maps_invalid_market_quantity_to_wait(fresh_obs):
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["BUY_SEED", "WHEAT", 0], ["BUY_SEED", "WHEAT"]],
    }

    normalized = D.normalize_expert_action(fresh_obs, action)

    assert normalized["market"] == [
        list(A.MARKET_WAIT_ACTION),
        list(A.MARKET_WAIT_ACTION),
    ]
    D.validate_policy_action(fresh_obs, normalized)


def test_normalize_expert_action_applies_atomic_plant_failure(fresh_obs):
    obs = copy.deepcopy(fresh_obs)
    obs["farms"][0]["farmer"] = [0, 0]
    obs["farms"][0]["hands"] = [[1, 0]]
    obs["private"]["inventories"] = [{}, {}]
    obs["private"]["seeds"]["STRAWBERRY"] = 1
    action = {
        "farmer": ["PLANT", "STRAWBERRY"],
        "hands": [["PLANT", "STRAWBERRY"]],
        "market": [],
    }

    normalized = D.normalize_expert_action(obs, action)

    assert normalized["farmer"] == ["PASS"]
    assert normalized["hands"] == [["PASS"]]
    D.validate_policy_action(obs, normalized)


def test_normalize_expert_action_keeps_animal_place_with_zero_quantity(fresh_obs):
    obs = copy.deepcopy(fresh_obs)
    x, y = obs["farms"][0]["farmer"]
    obs["farms"][0]["tiles"][y][x] = {"kind": "COOP"}
    obs["private"]["inventories"][0]["GOOSE"] = 1
    action = {"farmer": ["PLACE", "GOOSE", 0], "hands": [], "market": []}

    normalized = D.normalize_expert_action(obs, action)

    assert normalized["farmer"] == ["PLACE", "GOOSE"]
    D.validate_policy_action(obs, normalized)


def test_normalize_expert_action_maps_invalid_unit_action_to_pass(fresh_obs):
    """farmerは空きタイルにいるのでHARVESTは候補に無い→PASSへ正規化する。"""
    action = {"farmer": ["HARVEST"], "hands": [], "market": []}
    normalized = D.normalize_expert_action(fresh_obs, action)
    assert normalized["farmer"] == ["PASS"]
    D.validate_policy_action(fresh_obs, normalized)


def test_normalize_expert_action_maps_invalid_market_order_to_wait(fresh_obs):
    """shed空のday0局面でSELL WHEATは候補に無い→後続の有効な注文の位置を保つ
    ため、STOPではなくWAITへ正規化する。"""
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["SELL", "WHEAT", 1], ["BUY_LAND"]],
    }
    normalized = D.normalize_expert_action(fresh_obs, action)
    assert normalized["market"][0] == list(A.MARKET_WAIT_ACTION)
    assert normalized["market"][1] == ["BUY_LAND"]  # 後続の有効な注文は保たれる
    D.validate_policy_action(fresh_obs, normalized)


def test_normalize_expert_action_clamps_over_limit_quantity(fresh_obs):
    """(op, item)自体は合法だが、要求数量がshed在庫を超える場合は上限にクランプする
    (候補に無い扱いにしてWAITへは正規化しない)。"""
    obs = copy.deepcopy(fresh_obs)
    obs["private"]["shed"]["WHEAT"] = 5
    action = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 999]]}

    normalized = D.normalize_expert_action(obs, action)

    assert normalized["market"] == [["SELL", "WHEAT", 5]]
    D.validate_policy_action(obs, normalized)


def test_normalize_expert_action_never_raises_on_real_replay_mismatches():
    """このセッションで確認済みの、正規化前は無効だった実リプレイの1ターンでも
    正規化後は必ず表現可能になることを確認する(_NormalizingChooserは常に
    PASS/MARKET_WAIT/クランプ済み数量のいずれかを選ぶため失敗しない)。
    """
    obs = {
        "day": 5,
        "hour": 4,
        "player": 0,
        "farms": [
            {
                "farmer": [4, 4],
                "hands": [],
                "hires_today": 0,
                "money": 100.0,
                "tiles": [[None] * 10 for _ in range(10)],
                "unlocked_quadrants": ["NW"],
            },
            {
                "farmer": [4, 4],
                "hands": [],
                "hires_today": 0,
                "money": 100.0,
                "tiles": [[None] * 10 for _ in range(10)],
                "unlocked_quadrants": ["NW"],
            },
        ],
        "private": {
            "inventories": [{}],
            "seeds": dict.fromkeys(C.CROPS, 0),
            "shed": dict.fromkeys(C.SHED_ITEMS, 0),
        },
        "market": {
            "inventory": dict.fromkeys(C.PRODUCTS, 10_000),
            "prices": dict.fromkeys(C.PRODUCTS, 100),
        },
        "town": {"unlocked_shops": []},
        "remainingOverageTime": 60,
    }
    # 所持金不足でBUY_SEED不可、shed空でSELL不可という、明らかに無効な要求だけの行動。
    action = {
        "farmer": ["HARVEST"],
        "hands": [],
        "market": [["BUY_SEED", "MELON", 100], ["SELL", "WHEAT", 50]],
    }

    normalized = D.normalize_expert_action(obs, action)

    D.validate_policy_action(obs, normalized)


def test_act_batch_value_is_1d(net, fresh_obs):
    """net.value()は(batch,1)を返すが、act_batch/evaluate_actions_batchは
    (batch,)に潰して返す契約(distribution._evaluate_values参照)。ここが崩れると
    (values - returns)がPPOのvalue lossで誤ってbroadcastされ黙って壊れる。
    """
    _, values, _, _, _ = D.act_batch(net, [fresh_obs, fresh_obs, fresh_obs])
    assert values.shape == (3,)


def test_evaluate_actions_batch_value_is_1d(net, fresh_obs):
    action, *_ = D.act(net, fresh_obs)
    values, _, _, _ = D.evaluate_actions_batch(net, [(fresh_obs, action)] * 3)
    assert values.shape == (3,)


def test_act_evaluate_consistency(net, fresh_obs):
    """act()でサンプリングした行動をevaluate_actions()で再評価すると、同じ
    value/log_prob/entropyになる(PPOのratio計算が成立するための前提)。
    """
    action, value, log_prob, entropy, nd = D.act(net, fresh_obs)
    value2, log_prob2, entropy2, nd2 = D.evaluate_actions(net, fresh_obs, action)
    assert nd == nd2
    assert torch.allclose(value, value2, atol=1e-4)
    assert torch.allclose(log_prob, log_prob2, atol=1e-4)
    assert torch.allclose(entropy, entropy2, atol=1e-4)


def test_market_wait_consumes_slot_and_continues(net, fresh_obs):
    """市場WAITは数量スロットを作らず、次の市場opスロットへ進む。"""
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [list(A.MARKET_WAIT_ACTION)],
    }

    log_prob, entropy, num_decisions = D.evaluate_policy(net, fresh_obs, action)

    assert torch.isfinite(log_prob)
    assert torch.isfinite(entropy)
    assert num_decisions == 3  # farmer PASS、market WAIT、market STOP


def test_market_wait_serializes_as_zero_quantity_order(fresh_obs):
    """内部WAIT候補はKaggle環境で1スロットだけ消費する数量0注文へ変換する。"""
    env = D._envs_from_observations([fresh_obs], 24, 100, 1, C.MAX_MARKET_ORDERS)[0]
    gen = D._decode_turn_gen(env)

    unit_cands, *_ = next(gen)
    pass_token = V.ACTION_FARMER_OP[C.FARMER_OP_NAMES.index("PASS")]
    pass_idx = next(i for i, cand in enumerate(unit_cands) if pass_token in cand.index)

    market_cands, *_ = gen.send(pass_idx)
    wait_idx = next(
        i for i, cand in enumerate(market_cands) if V.ACTION_MARKET_WAIT[0] in cand.index
    )
    next_market_cands, *_ = gen.send(wait_idx)
    stop_idx = next(
        i for i, cand in enumerate(next_market_cands) if V.ACTION_MARKET_STOP[0] in cand.index
    )

    with pytest.raises(StopIteration) as done:
        gen.send(stop_idx)
    assert done.value.value["market"] == [list(A.MARKET_WAIT_ACTION)]


def test_batch_matches_single(net, fresh_obs):
    """act_batchの結果が、1件ずつevaluate_actionsした結果と一致する
    (バッチ化してもcausal maskの数学的な等価性で1件ずつの計算と同一になるはず)。
    """
    obs_list = [fresh_obs] * 5
    actions, values, log_probs, entropies, nds = D.act_batch(net, obs_list)
    for i in range(5):
        v, lp, ent, nd = D.evaluate_actions(net, fresh_obs, actions[i])
        assert torch.allclose(v, values[i], atol=1e-4)
        assert torch.allclose(lp, log_probs[i], atol=1e-4)
        assert torch.allclose(ent, entropies[i], atol=1e-4)
        assert nd == nds[i]


def test_ppo_loss_gradients_are_finite(net, fresh_obs):
    """value/policy/entropyを含むPPO風lossをbackwardして、全パラメータの勾配が
    finiteであることを確認する。entropy係数0でも確認する(padding候補のentropy
    項が0*(-inf)=nanになる形の壊れ方は、backward自体は必ず実行されるため係数が
    0でも顕在化する。実際に過去に一度この形でNaNが発生した)。
    """
    pairs = []
    for _ in range(4):
        action, *_ = D.act(net, fresh_obs)
        pairs.append((fresh_obs, action))

    for entropy_coef in (0.0, 0.01):
        net.zero_grad()
        values, log_probs, entropies, _ = D.evaluate_actions_batch(net, pairs)
        returns = torch.zeros_like(values)
        loss = (
            (values - returns).square().mean() - log_probs.mean() - entropy_coef * entropies.mean()
        )
        loss.backward()
        for name, p in net.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert torch.isfinite(p.grad).all(), (
                f"{name} has non-finite grad (entropy_coef={entropy_coef})"
            )


def test_masked_entropy_backward_is_finite(net):
    """_masked_entropyの回帰テスト: paddingされた候補(候補数が行ごとに異なり、
    一部の行にしか存在しない列)がある状態でentropyをbackwardしても、
    0*(-inf)=nanが勾配に残らないことを直接確認する。
    """
    hidden = torch.randn(2, net.decoder.d_model, requires_grad=True)
    cands_per_row = [
        [V.SparseVector([i], [1.0]) for i in range(3)],
        [V.SparseVector([0], [1.0])],  # 候補1つだけ(他の行よりpaddingが多く発生)
    ]
    scores = net.decoder.score_candidates_batch(hidden, cands_per_row)
    log_probs = torch.log_softmax(scores, dim=-1)
    probs = log_probs.exp()
    entropy = D._masked_entropy(scores, log_probs, probs)
    entropy.sum().backward()

    assert torch.isfinite(hidden.grad).all()
    for p in net.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_score_candidates_grouped_matches_single(net):
    """_score_candidates_grouped(slot_kind×候補数バケットごとにpaddingを分ける
    バッチ化)が、候補数の少ないslotと多いslot(quantityで最大100)が混在して
    いても、score_candidates(単一行)を個別に呼んだ場合と同じ結果になることを
    確認する。quantity同士でも候補数が離れていれば別バケットに分かれる
    (_padding_group_key参照)ことも合わせて検証する。
    """
    hidden = torch.randn(5, net.decoder.d_model)
    cands_per_row = [
        [V.SparseVector([i], [1.0]) for i in range(5)],
        [V.SparseVector([i], [1.0]) for i in range(100)],
        [V.SparseVector([i], [1.0]) for i in range(8)],
        [V.SparseVector([i], [1.0]) for i in range(3)],  # quantity、小バケット
        [V.SparseVector([i], [1.0]) for i in range(50)],  # quantity、中バケット
    ]
    slot_kinds = ["unit_op", "quantity", "market_op", "quantity", "quantity"]

    grouped = D._score_candidates_grouped(net, hidden, cands_per_row, slot_kinds)

    for row, cands in enumerate(cands_per_row):
        single = net.decoder.score_candidates(hidden[row], cands)
        assert torch.allclose(grouped[row, : len(cands)], single, atol=1e-5)
        if len(cands) < grouped.shape[1]:
            assert torch.isinf(grouped[row, len(cands) :]).all()
