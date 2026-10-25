import math

import pytest
import torch

from dms.distributions import categorical_action, gaussian_action, masked_categorical


def test_mask_excludes_invalid_samples_and_deterministic_actions():
    logits = torch.tensor([1.0, 100.0, 2.0, 200.0]).expand(4096, -1)
    mask = torch.tensor([True, False, True, False])
    actions, _, _ = categorical_action(logits, mask)
    assert torch.all((actions == 0) | (actions == 2))
    deterministic, _, _ = categorical_action(logits, mask, deterministic=True)
    assert torch.all(deterministic == 2)


def test_categorical_collection_and_update_logprobs_are_identical():
    logits = torch.tensor([[[0.2, -1.0, 2.0], [1.0, 0.0, -2.0]]])
    mask = torch.tensor([[[True, True, False], [True, False, True]]])
    actions, old_logprob, old_entropy = categorical_action(logits, mask)
    evaluated, logprob, entropy = categorical_action(logits, mask, actions=actions)
    assert actions.shape == (1, 2)
    torch.testing.assert_close(evaluated, actions)
    torch.testing.assert_close(logprob, old_logprob, rtol=0, atol=0)
    torch.testing.assert_close(entropy, old_entropy, rtol=0, atol=0)
    torch.testing.assert_close(torch.exp(logprob - old_logprob), torch.ones_like(logprob))


def test_masked_entropy_and_gradients_are_finite():
    logits = torch.zeros(3, requires_grad=True)
    mask = torch.tensor([True, False, True])
    _, logprob, entropy = categorical_action(logits, mask, actions=torch.tensor(0))
    torch.testing.assert_close(entropy, torch.tensor(math.log(2.0)))
    torch.testing.assert_close(logprob, torch.tensor(-math.log(2.0)))
    (logprob + entropy).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1] == 0


def test_empty_available_action_set_is_an_error():
    with pytest.raises(ValueError, match="at least one available"):
        masked_categorical(torch.zeros(2, 3), torch.tensor([[True, False, False], [False, False, False]]))


def test_gaussian_matches_hand_computed_joint_density_and_entropy():
    mean = torch.tensor([[1.0, -2.0], [0.0, 3.0]], dtype=torch.float64)
    log_std = torch.log(torch.tensor([2.0, 0.5], dtype=torch.float64))
    actions = torch.tensor([[3.0, -2.5], [0.0, 3.0]], dtype=torch.float64)
    evaluated, logprob, entropy = gaussian_action(mean, log_std, actions=actions)
    # std products are 1, so joint log-density at the mean is -log(2*pi).
    expected_logprob = torch.tensor([-1.0 - math.log(2 * math.pi), -math.log(2 * math.pi)], dtype=torch.float64)
    expected_entropy = torch.full((2,), 1.0 + math.log(2 * math.pi), dtype=torch.float64)
    torch.testing.assert_close(evaluated, actions)
    torch.testing.assert_close(logprob, expected_logprob)
    torch.testing.assert_close(entropy, expected_entropy)


def test_gaussian_collection_and_update_logprobs_are_identical():
    mean = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    log_std = torch.tensor([-1.0, 0.0, 0.5, 1.0])
    actions, old_logprob, old_entropy = gaussian_action(mean, log_std)
    _, logprob, entropy = gaussian_action(mean, log_std, actions=actions)
    deterministic, _, _ = gaussian_action(mean, log_std, deterministic=True)
    assert logprob.shape == (2, 3)
    torch.testing.assert_close(logprob, old_logprob, rtol=0, atol=0)
    torch.testing.assert_close(entropy, old_entropy, rtol=0, atol=0)
    torch.testing.assert_close(deterministic, mean, rtol=0, atol=0)
