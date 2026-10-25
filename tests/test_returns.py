import pytest
import torch

from dms.returns import generalized_advantage_estimate


def test_time_limit_bootstraps_but_neither_boundary_leaks_next_episode():
    # Agent 0 hits a timeout at t=1; agent 1 genuinely terminates there.
    rewards = torch.tensor([1.0, 2.0, 3.0, 8.0], dtype=torch.float64)[:, None, None].expand(-1, 1, 2)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)[:, None, None].expand(-1, 1, 2)
    successors = torch.tensor([2.0, 10.0, 4.0, float("nan")], dtype=torch.float64)[:, None, None].expand(-1, 1, 2)
    terminated = torch.zeros_like(rewards, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    truncated[1, 0, 0] = True
    terminated[1, 0, 1] = True
    terminated[3] = True

    advantages, returns = generalized_advantage_estimate(
        rewards, values, successors, terminated, truncated, gamma=0.9, gae_lambda=0.5
    )

    # At t=1: timeout delta = 2 + .9*10 - 2 = 9; terminal delta = 0.
    # t=2: 3 + .9*4 - 3 + .45*(8-4) = 5.4.
    expected = torch.tensor([[5.85, 1.8], [9.0, 0.0], [5.4, 5.4], [4.0, 4.0]], dtype=torch.float64)[:, None]
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected + values)


def test_final_rollout_step_bootstraps_without_episode_boundary():
    rewards = torch.tensor([[[2.0]]])
    values = torch.tensor([[[3.0]]], requires_grad=True)
    successors = torch.tensor([[[5.0]]], requires_grad=True)
    boundary = torch.zeros_like(rewards, dtype=torch.bool)
    advantages, returns = generalized_advantage_estimate(
        rewards, values, successors, boundary, boundary, gamma=0.8
    )
    torch.testing.assert_close(advantages, torch.tensor([[[3.0]]]))
    torch.testing.assert_close(returns, torch.tensor([[[6.0]]]))
    assert not advantages.requires_grad
    assert not returns.requires_grad


def test_zero_lambda_gives_one_step_targets():
    rewards = torch.tensor([1.0, 2.0])[:, None, None]
    values = torch.tensor([4.0, 7.0])[:, None, None]
    successors = torch.tensor([7.0, 10.0])[:, None, None]
    boundary = torch.zeros_like(rewards, dtype=torch.bool)
    _, returns = generalized_advantage_estimate(
        rewards, values, successors, boundary, boundary, gamma=0.5, gae_lambda=0.0
    )
    torch.testing.assert_close(returns, torch.tensor([4.5, 7.0])[:, None, None])


def test_rejects_ambiguous_boundary_broadcasting():
    x = torch.zeros(3, 2, 4)
    with pytest.raises(ValueError, match="terminated must have shape"):
        generalized_advantage_estimate(x, x, x, torch.zeros(3, 2, dtype=torch.bool), x.bool())
    with pytest.raises(TypeError, match="boolean"):
        generalized_advantage_estimate(x, x, x, x, x)
