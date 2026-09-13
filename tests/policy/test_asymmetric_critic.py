"""非対称critic(model.PolicyValueNet(use_asymmetric_critic=True))の契約テスト。
実データに依存せず、合成observation(conftest.fresh_obs/fresh_private)だけで
完結する。
"""

import copy

import torch

from kaggriculture.policy.common import layout as L
from kaggriculture.policy.common import vocab as V
from kaggriculture.policy.torch import distribution as D
from kaggriculture.policy.torch import tokenize


def test_opponent_private_none_is_rejected(net_with_asymmetric_critic, fresh_obs):
    """use_asymmetric_critic=Trueのネットにopponent_privates=None(省略)で
    呼ぶとValueErrorになる(actor用の入力しか無い状態でcriticを評価できない
    ことを黙って無視しない)。"""
    try:
        D.act(net_with_asymmetric_critic, fresh_obs)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_opponent_private_rejected_when_not_using_asymmetric_critic(net, fresh_obs, fresh_private):
    """use_asymmetric_critic=False(既定)のネットにopponent_privateを渡すと
    エラーになる(黙って無視されない)。"""
    try:
        D.act(net, fresh_obs, opponent_private=fresh_private)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_asymmetric_critic_does_not_affect_action_distribution(
    net_with_asymmetric_critic, fresh_obs, fresh_private
):
    """固定した重みでの1回の順伝播では、opponent_privateの中身を変えても
    actor(Decoderが選ぶ候補・log_prob)は変わらないことを確認する(critic_
    encoderへの入力だけが変わり、self.decoderは一切参照しないため)。

    これは「学習が進んでもactorの表現にcriticが一切影響しない」ことまでは
    意味しない。token_embedding/board_position_embedding/self.encoderはcritic
    と共有しているため、value lossの逆伝播はこれらにも勾配を流す
    (model.PolicyValueNet.__init__のuse_asymmetric_critic docstring参照)。
    """
    net = net_with_asymmetric_critic
    torch.manual_seed(0)
    action1, value1, log_prob1, entropy1, nd1 = D.act(
        net, fresh_obs, opponent_private=fresh_private
    )

    opp_private_nonempty = dict(fresh_private)
    opp_private_nonempty["shed"] = dict(fresh_private["shed"])
    opp_private_nonempty["shed"]["WHEAT"] = 999

    torch.manual_seed(0)
    action2, value2, log_prob2, entropy2, nd2 = D.act(
        net, fresh_obs, opponent_private=opp_private_nonempty
    )

    assert action1 == action2
    assert nd1 == nd2
    assert abs(log_prob1.item() - log_prob2.item()) < 1e-6
    assert abs(entropy1.item() - entropy2.item()) < 1e-6
    # 一方でvalueはprivileged_encoderの入力が変わったので変化してよい(むしろ変化する
    # べき。死んだ引数になっていないことの確認)。
    assert abs(value1.item() - value2.item()) > 1e-9


def test_asymmetric_critic_act_evaluate_consistency(
    net_with_asymmetric_critic, fresh_obs, fresh_private
):
    net = net_with_asymmetric_critic
    action, value, log_prob, entropy, nd = D.act(net, fresh_obs, opponent_private=fresh_private)
    value2, log_prob2, entropy2, nd2 = D.evaluate_actions(
        net, fresh_obs, action, opponent_private=fresh_private
    )
    assert nd == nd2
    assert abs(value.item() - value2.item()) < 1e-4
    assert abs(log_prob.item() - log_prob2.item()) < 1e-4
    assert abs(entropy.item() - entropy2.item()) < 1e-4


def test_asymmetric_critic_gradients_are_finite(
    net_with_asymmetric_critic, fresh_obs, fresh_private
):
    net = net_with_asymmetric_critic
    net.train()
    action, *_ = D.act(net, fresh_obs, opponent_private=fresh_private)
    value, log_prob, entropy, _nd = D.evaluate_actions(
        net, fresh_obs, action, opponent_private=fresh_private
    )
    loss = -log_prob - 0.01 * entropy + value.square()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite grad"


def test_privileged_tokens_bind_inventory_to_unit_position(fresh_obs, fresh_private):
    obs = copy.deepcopy(fresh_obs)
    opponent = 1 - obs["player"]
    obs["farms"][opponent]["hands"] = [[0, 1]]
    opponent_private = copy.deepcopy(fresh_private)
    opponent_private["inventories"] = [{}, {"WHEAT": 3}]

    tokens, positions, padding = tokenize.get_privileged_critic_input(obs, opponent_private)

    opponent_base = L.PRIVILEGED_TOKENS_PER_OWNER
    opponent_farmer = opponent_base + 2
    opponent_hand0 = opponent_farmer + 1
    fx, fy = obs["farms"][opponent]["farmer"]
    assert positions[opponent_farmer] == fy * V.BOARD_SIZE + fx
    assert positions[opponent_hand0] == V.BOARD_SIZE
    assert V.entity_index("WHEAT") in tokens[opponent_hand0].index
    assert not padding[opponent_hand0]
    assert padding[opponent_hand0 + 1]


def test_critic_value_distinguishes_which_unit_holds_item(
    net_with_asymmetric_critic, fresh_obs, fresh_private
):
    obs = copy.deepcopy(fresh_obs)
    opponent = 1 - obs["player"]
    obs["farms"][opponent]["hands"] = [[0, 1]]

    farmer_holds = copy.deepcopy(fresh_private)
    farmer_holds["inventories"] = [{"WHEAT": 3}, {}]
    hand_holds = copy.deepcopy(fresh_private)
    hand_holds["inventories"] = [{}, {"WHEAT": 3}]

    torch.manual_seed(0)
    action1, value1, log_prob1, *_ = D.act(
        net_with_asymmetric_critic, obs, opponent_private=farmer_holds
    )
    torch.manual_seed(0)
    action2, value2, log_prob2, *_ = D.act(
        net_with_asymmetric_critic, obs, opponent_private=hand_holds
    )

    assert action1 == action2
    assert torch.allclose(log_prob1, log_prob2)
    assert not torch.allclose(value1, value2)


def test_privileged_branch_is_shorter_and_shallower(net_with_asymmetric_critic):
    critic = net_with_asymmetric_critic.privileged_encoder
    assert L.NUM_PRIVILEGED_TOKENS < L.NUM_WORDS_ENCODER
    assert len(critic.transformer.layers) < len(
        net_with_asymmetric_critic.encoder.transformer.layers
    )


def test_predict_action_is_actor_only(net_with_asymmetric_critic, fresh_obs, fresh_private):
    torch.manual_seed(0)
    expected, *_ = D.act(
        net_with_asymmetric_critic,
        fresh_obs,
        opponent_private=fresh_private,
    )
    torch.manual_seed(0)
    actual = D.predict_action(net_with_asymmetric_critic, fresh_obs)
    assert actual == expected
