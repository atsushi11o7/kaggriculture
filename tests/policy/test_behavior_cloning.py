"""BC用actor-only教師強制APIの契約テスト。"""

import torch

from kaggriculture.policy import distribution as D
from kaggriculture.training.bc import mean_token_nll


def test_policy_evaluation_skips_critic(net_with_asymmetric_critic, fresh_obs):
    net = net_with_asymmetric_critic
    action = D.predict_action(net, fresh_obs)
    net.train()
    net.zero_grad()
    calls = 0

    def count_call(_module, _args, _output):
        nonlocal calls
        calls += 1

    handle = net.privileged_encoder.register_forward_hook(count_call)
    try:
        log_probs, _, num_decisions = D.evaluate_policy_batch(net, [(fresh_obs, action)])
        mean_token_nll(log_probs, num_decisions).backward()
    finally:
        handle.remove()

    assert calls == 0
    assert any(p.grad is not None for p in net.decoder.parameters())
    assert all(p.grad is None for p in net.value_head.parameters())
    for name, parameter in net.privileged_encoder.named_parameters():
        if not name.startswith(("token_embedding.", "position_embedding.")):
            assert parameter.grad is None


def test_policy_evaluation_matches_ppo_path(net, fresh_obs):
    action = D.predict_action(net, fresh_obs)
    net.eval()

    log_prob, entropy, num_decisions = D.evaluate_policy(net, fresh_obs, action)
    _, ppo_log_prob, ppo_entropy, ppo_num_decisions = D.evaluate_actions(net, fresh_obs, action)

    assert torch.allclose(log_prob, ppo_log_prob)
    assert torch.allclose(entropy, ppo_entropy)
    assert num_decisions == ppo_num_decisions


def test_mean_token_nll_uses_total_decision_count():
    log_probs = torch.tensor([-2.0, -6.0])

    loss = mean_token_nll(log_probs, [2, 3])

    assert torch.allclose(loss, torch.tensor(1.6))
