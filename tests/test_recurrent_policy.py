import copy

import pytest
import torch
from torch.distributions import kl_divergence

from dms.policy import IndependentAgent, RunningMoments


def inputs(batch=6, length=4, width=3, peers=3, latent=2):
    base = torch.randn(batch, length, width)
    valid = torch.arange(length)[None] >= torch.arange(batch)[:, None] % (length+1)
    windows = torch.cat((base, valid[..., None].float()), dim=-1)
    context = torch.randn(batch, peers, latent)
    gates = torch.ones(batch, peers, dtype=torch.bool); gates[:, 0] = False
    return windows, context, gates


def test_default_checkpoint_keys_weights_and_feedforward_behavior_are_unchanged():
    torch.manual_seed(20)
    default = IndependentAgent(3, 4, 3, 2, discrete=True)
    torch.manual_seed(20)
    explicit = IndependentAgent(3, 4, 3, 2, discrete=True, policy_history_length=0)
    assert not hasattr(default, 'actor_encoder') and not hasattr(default, 'critic_encoder')
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key in default.state_dict():
        torch.testing.assert_close(default.state_dict()[key], explicit.state_dict()[key], atol=0, rtol=0)
    obs, context = torch.randn(5, 3), torch.randn(5, 3, 2)
    gates = torch.ones(5, 3, dtype=torch.bool)
    expected_logits = default.actor(default._inputs(obs, context, gates))
    actual = default.distribution(obs, context, gates)
    expected_log_prob = expected_logits - expected_logits.logsumexp(-1, keepdim=True)
    torch.testing.assert_close(actual.logits, expected_log_prob, atol=0, rtol=0)
    torch.testing.assert_close(default.normalized_value(obs, context, gates),
                               default.critic(default._inputs(obs, context, gates)).squeeze(-1), atol=0, rtol=0)
    old_moments = copy.deepcopy(default.obs_moments)
    old_moments.update(obs)
    default.update_observation_scale(obs)
    torch.testing.assert_close(default.obs_moments.mean, old_moments.mean, atol=0, rtol=0)
    assert list(default.critic_parameters()) == list(default.critic.parameters())


@pytest.mark.parametrize('coordination', ['soft', 'causal', 'all'])
def test_recurrent_masked_likelihood_replay_and_checkpoint_roundtrip(coordination):
    torch.manual_seed(21)
    agent = IndependentAgent(3, 4, 3, 2, discrete=True, coordination=coordination,
                             policy_history_length=4)
    windows, context, gates = inputs()
    available = torch.tensor([[True, False, True, True]]).expand(6, -1)
    normalized = agent.normalize_obs(windows)
    action, old_log_prob, values = agent.act(normalized, context, gates, available)
    order = torch.tensor([5, 0, 3, 1, 4, 2])
    replay = agent.distribution(normalized[order], context[order], gates[order], available[order])
    torch.testing.assert_close(replay.log_prob(action[order]), old_log_prob[order], atol=2e-6, rtol=1e-6)
    assert not (action == 1).any() and torch.equal(replay.probs[:, 1], torch.zeros(6))
    restored = IndependentAgent(3, 4, 3, 2, discrete=True, coordination=coordination,
                                policy_history_length=4)
    restored.load_state_dict(copy.deepcopy(agent.state_dict()))
    torch.testing.assert_close(restored.value(normalized, context, gates), values, atol=0, rtol=0)
    assert agent.obs_moments.mean.shape == (3,)
    torch.testing.assert_close(normalized[..., -1], windows[..., -1], atol=0, rtol=0)


def test_history_moments_count_only_latest_valid_observations_once():
    agent = IndependentAgent(3, 2, 2, policy_history_length=4)
    raw = torch.full((2, 3, 4, 4), 1000.)
    raw[..., -1] = 1
    newest = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    raw[..., -1, :3] = newest
    raw[0, 1, :, -1] = 0  # one dead/padded stream
    expected = RunningMoments(3)
    expected.update(newest[torch.tensor([[True, False, True], [True, True, True]])])
    agent.update_observation_scale(raw)
    for key in expected.state_dict():
        torch.testing.assert_close(agent.obs_moments.state_dict()[key], expected.state_dict()[key], atol=0, rtol=0)
    before = copy.deepcopy(agent.obs_moments.state_dict())
    raw[..., -1] = 0
    agent.update_observation_scale(raw)
    for key in before:
        torch.testing.assert_close(agent.obs_moments.state_dict()[key], before[key], atol=0, rtol=0)


@pytest.mark.parametrize('invalid', [.5, -1., 2., float('nan')])
def test_validity_channel_is_never_implicitly_normalized_or_rounded(invalid):
    agent = IndependentAgent(3, 2, 3, 2, policy_history_length=4)
    windows, context, gates = inputs()
    windows[0, 0, -1] = invalid
    with pytest.raises(ValueError, match='exactly 0 or 1'):
        agent.normalize_obs(windows)
    with pytest.raises(ValueError, match='exactly 0 or 1'):
        agent.distribution(windows, context, gates)


def test_actor_and_critic_encoders_have_separate_gradients_and_optimizer_ownership():
    torch.manual_seed(22)
    agent = IndependentAgent(3, 4, 3, 2, discrete=True, policy_history_length=4)
    windows, context, gates = inputs()
    dist = agent.distribution(windows, context, gates)
    dist.log_prob(torch.zeros(6, dtype=torch.long)).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor_encoder.parameters())
    assert all(p.grad is None for p in agent.critic_parameters())
    agent.zero_grad(set_to_none=True)
    agent.normalized_value(windows, context, gates).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.critic_encoder.parameters())
    assert all(p.grad is None for p in agent.actor_encoder.parameters())
    critic_ids = {id(p) for p in agent.critic_parameters()}
    expected = {id(p) for p in agent.critic.parameters()} | {id(p) for p in agent.critic_encoder.parameters()}
    assert critic_ids == expected
    assert not critic_ids & {id(p) for p in agent.actor_encoder.parameters()}


def test_popart_preserves_recurrent_raw_values():
    torch.manual_seed(23)
    agent = IndependentAgent(3, 2, 3, 2, policy_history_length=4)
    windows, context, gates = inputs()
    before = agent.value(windows, context, gates).detach()
    agent.update_value_scale(torch.linspace(-100, 300, 30))
    torch.testing.assert_close(agent.value(windows, context, gates), before, atol=4e-5, rtol=4e-5)


def test_causal_relevance_encodes_local_history_once_and_only_ablates_context(monkeypatch):
    torch.manual_seed(24)
    agent = IndependentAgent(3, 4, 3, 2, discrete=True, coordination='causal', policy_history_length=4)
    windows, context, visible = inputs()
    mask = torch.tensor([[True, False, True, True]]).expand(6, -1)
    full = agent.distribution(windows, context, visible, mask)
    expected = torch.zeros_like(visible)
    for peer in range(3):
        removed = visible.clone(); removed[:, peer] = False
        other = agent.distribution(windows, context, removed, mask)
        expected[:, peer] = visible[:, peer] & (kl_divergence(full, other) > 1e-8)
    calls = []
    original = agent.actor_encoder.encode_windows

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(agent.actor_encoder, 'encode_windows', counted)
    actual = agent.relevance(windows, context, visible, mask, threshold=1e-8)
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected)


def test_padding_and_invisible_context_cannot_change_recurrent_policy():
    torch.manual_seed(25)
    agent = IndependentAgent(3, 4, 3, 2, discrete=True, policy_history_length=4)
    windows, context, gates = inputs()
    gates[:, 2] = False
    before = agent.distribution(windows, context, gates).probs.detach()
    changed = windows.clone(); changed[..., :3][~windows[..., -1].bool()] = float('nan')
    context[:, 2] += 1000
    torch.testing.assert_close(agent.distribution(changed, context, gates).probs, before, atol=0, rtol=0)


def test_history_without_intentions_supports_continuous_actions():
    agent = IndependentAgent(3, 2, 3, policy_history_length=4)
    windows, context, gates = inputs(latent=0)
    action, likelihood, value = agent.act(windows, context, gates)
    assert action.shape == (6, 2) and value.shape == likelihood.shape == (6,)
    torch.testing.assert_close(agent.distribution(windows, context, gates).log_prob(action), likelihood)
    assert torch.isfinite(value).all()
