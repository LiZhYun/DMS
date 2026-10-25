import torch
from dms.policy import IndependentAgent


def test_popart_preserves_raw_value():
    torch.manual_seed(1)
    agent = IndependentAgent(4, 2, 3, latent_dim=2)
    obs = torch.randn(7, 4)
    intentions = torch.randn(7, 3, 2)
    relevant = torch.ones(7, 3, dtype=torch.bool)
    before = agent.value(obs, intentions, relevant).detach()
    agent.update_value_scale(torch.linspace(100, 300, 100))
    torch.testing.assert_close(before, agent.value(obs, intentions, relevant), atol=3e-5, rtol=3e-5)


def test_causal_mask_and_cached_likelihood():
    torch.manual_seed(3)
    agent = IndependentAgent(4, 3, 2, 2, discrete=True, coordination='causal')
    obs, intentions = torch.randn(6, 4), torch.randn(6, 2, 2)
    visible = torch.tensor([[True, False]]).expand(6, -1)
    actions_available = torch.tensor([[True, False, True]]).expand(6, -1)
    relevant = agent.relevance(obs, intentions, visible, actions_available, threshold=0)
    assert not relevant.any()  # self is a query, never its own coordination edge
    action, old_log_prob, _ = agent.act(obs, intentions, relevant, actions_available)
    new_log_prob = agent.distribution(obs, intentions, relevant, actions_available).log_prob(action)
    torch.testing.assert_close((new_log_prob-old_log_prob).exp(), torch.ones(6))


def test_invisible_intention_cannot_change_policy():
    agent = IndependentAgent(4, 2, 3, 2, discrete=True)
    obs, intentions = torch.randn(5, 4), torch.randn(5, 3, 2)
    mask = torch.tensor([[True, True, False]]).expand(5, -1)
    before = agent.distribution(obs, intentions, mask).probs.detach()
    intentions[:, 2] += 100
    torch.testing.assert_close(before, agent.distribution(obs, intentions, mask).probs)
