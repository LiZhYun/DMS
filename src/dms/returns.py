"""Time-limit-aware generalized advantage estimation.

``next_values[t]`` must value the actual successor observation, including the
final observation before an environment auto-reset. A reset observation is not
a valid bootstrap target for a truncated episode.
"""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Return detached ``(advantages, returns)`` with shape ``[time, env, agent]``.

    True terminations suppress the one-step value bootstrap. Time limits retain
    that bootstrap. Both boundaries stop advantage recursion, so experience
    from the next episode never leaks into the previous one. The last timestep
    also bootstraps using ``next_values`` but has no subsequent residual.
    """
    if rewards.ndim != 3 or rewards.shape[0] == 0:
        raise ValueError("rewards must have nonempty shape [time, env, agent]")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    for name, tensor in (
        ("values", values),
        ("next_values", next_values),
        ("terminated", terminated),
        ("truncated", truncated),
    ):
        if tensor.shape != rewards.shape:
            raise ValueError(f"{name} must have shape {tuple(rewards.shape)}")
        if tensor.device != rewards.device:
            raise ValueError(f"{name} must be on the same device as rewards")
    if not rewards.is_floating_point():
        raise TypeError("rewards must have a floating-point dtype")
    if values.dtype != rewards.dtype or next_values.dtype != rewards.dtype:
        raise TypeError("rewards, values, and next_values must have the same dtype")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise TypeError("terminated and truncated must be boolean tensors")

    # where avoids 0 * nan at terminal states whose successor has no value.
    bootstrap_values = torch.where(terminated, 0.0, next_values)
    residuals = rewards + gamma * bootstrap_values - values
    continuation = ~(terminated | truncated)
    advantages = torch.empty_like(rewards)
    accumulated = torch.zeros_like(rewards[0])
    for time in reversed(range(rewards.shape[0])):
        accumulated = residuals[time] + gamma * gae_lambda * torch.where(
            continuation[time], accumulated, 0.0
        )
        advantages[time] = accumulated
    return advantages, advantages + values
