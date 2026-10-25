import math

import pytest
import torch
from torch.nn.functional import pad

from dms.policy import IndependentAgent


def policy_inputs(batch=5):
    return torch.randn(batch, 4), torch.zeros(batch, 2, 0), torch.zeros(batch, 2, dtype=torch.bool)


def test_default_scale_preserves_weights_checkpoint_keys_and_sampled_likelihoods():
    torch.manual_seed(41)
    default = IndependentAgent(4, 3, 2)
    torch.manual_seed(41)
    explicit = IndependentAgent(4, 3, 2, initial_log_std=-0.5)
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert not any('initial_log_std' in key for key in default.state_dict())
    for name, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], atol=0, rtol=0)
    torch.testing.assert_close(default.log_std, torch.full((3,), -0.5), atol=0, rtol=0)
    inputs = policy_inputs()
    torch.manual_seed(42)
    before = default.act(*inputs)
    torch.manual_seed(42)
    after = explicit.act(*inputs)
    for old, new in zip(before, after):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
    torch.testing.assert_close(default.distribution(*inputs).stddev, torch.full((5, 3), math.exp(-0.5)))


@pytest.mark.parametrize('log_std', [-5, -1, 2])
def test_configured_scale_matches_gaussian_standard_deviation_and_entropy(log_std):
    agent = IndependentAgent(4, 3, 2, initial_log_std=log_std)
    distribution = agent.distribution(*policy_inputs())
    assert agent.log_std.is_floating_point() and agent.log_std.requires_grad
    torch.testing.assert_close(distribution.stddev, torch.full((5, 3), math.exp(log_std)))
    expected_entropy = 3 * (log_std + .5 * math.log(2*math.pi*math.e))
    torch.testing.assert_close(distribution.entropy(), torch.full((5,), expected_entropy))


def test_heterogeneous_action_dimensions_never_add_padded_scale_parameters():
    agents = [IndependentAgent(4, width, 2, initial_log_std=-1) for width in (1, 3)]
    inputs = policy_inputs()
    results = [agent.act(*inputs) for agent in agents]
    padded = torch.stack([pad(result[0], (0, 3-result[0].shape[-1])) for result in results], dim=1)
    assert padded.shape == (5, 2, 3) and padded[:, 0, 1:].eq(0).all()
    for observer, width in enumerate((1, 3)):
        agent = agents[observer]
        assert agent.log_std.shape == (width,)
        distribution = agent.distribution(*inputs)
        torch.testing.assert_close(distribution.stddev, torch.full((5, width), math.exp(-1)))
        torch.testing.assert_close(distribution.log_prob(padded[:, observer, :width]), results[observer][1])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'),
                                  -5.001, 2.001, True, False, '-1', None])
def test_invalid_initial_scales_are_rejected(value):
    with pytest.raises(ValueError, match=r'finite number in \[-5, 2\]'):
        IndependentAgent(4, 3, 2, initial_log_std=value)


def test_discrete_weights_and_masked_distribution_ignore_valid_initial_scale_option():
    inputs = policy_inputs()
    mask = torch.tensor([[True, False, True]]).expand(5, -1)
    policies = []
    for value in (-5, -1, -.5, 2):
        torch.manual_seed(43)
        policies.append(IndependentAgent(4, 3, 2, discrete=True, initial_log_std=value))
    reference = policies[0]
    for agent in policies:
        assert agent.log_std is None and 'log_std' not in agent.state_dict()
        assert agent.state_dict().keys() == reference.state_dict().keys()
        for name, value in agent.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)
        torch.testing.assert_close(agent.distribution(*inputs, mask).probs,
                                   reference.distribution(*inputs, mask).probs, atol=0, rtol=0)


def test_loaded_learned_scale_overrides_constructor_initialization():
    trained = IndependentAgent(4, 3, 2)
    with torch.no_grad():
        trained.log_std.copy_(torch.tensor([-.8, -.6, -.4]))
    restored = IndependentAgent(4, 3, 2, initial_log_std=-1)
    restored.load_state_dict(trained.state_dict())
    torch.testing.assert_close(restored.log_std, trained.log_std, atol=0, rtol=0)
    inputs = policy_inputs()
    torch.testing.assert_close(restored.distribution(*inputs).stddev,
                               trained.distribution(*inputs).stddev, atol=0, rtol=0)
