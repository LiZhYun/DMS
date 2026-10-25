"""Small action-distribution helpers shared by collection and PPO updates."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical, Independent, Normal


def masked_categorical(logits: Tensor, action_mask: Tensor | None = None) -> Categorical:
    """Build a categorical distribution with exactly zero invalid-action mass.

    ``logits`` has shape ``[..., actions]``. A boolean mask may broadcast to that
    shape. Every batch item must have at least one available action; silently
    choosing an invalid action would hide an environment or collection bug.
    """
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a nonempty final action dimension")
    if action_mask is not None:
        if action_mask.dtype != torch.bool:
            raise TypeError("action_mask must be boolean")
        if action_mask.device != logits.device:
            raise ValueError("action_mask and logits must be on the same device")
        try:
            action_mask = torch.broadcast_to(action_mask, logits.shape)
        except RuntimeError as error:
            raise ValueError("action_mask must broadcast to logits.shape") from error
        if not action_mask.any(dim=-1).all().item():
            raise ValueError("each batch item must have at least one available action")
        logits = logits.masked_fill(~action_mask, -torch.inf)
    return Categorical(logits=logits)


def categorical_action(
    logits: Tensor,
    action_mask: Tensor | None = None,
    *,
    actions: Tensor | None = None,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample or evaluate an action and return ``(action, log_prob, entropy)``.

    All three outputs have shape ``logits.shape[:-1]``. Passing stored actions
    uses precisely the same distribution as collection, including its mask.
    """
    distribution = masked_categorical(logits, action_mask)
    if actions is None:
        actions = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
    elif actions.shape != logits.shape[:-1]:
        raise ValueError("actions must have shape logits.shape[:-1]")
    return actions, distribution.log_prob(actions), distribution.entropy()


def diagonal_gaussian(mean: Tensor, log_std: Tensor) -> Independent:
    """Build an unsquashed Gaussian, summing probabilities over action dimensions.

    ``mean`` has shape ``[..., actions]``; ``log_std`` can broadcast to that shape.
    Bounded environments should clip only the action sent to the environment and
    store the original sample for PPO. No tanh or clipping is hidden here.
    """
    if mean.ndim < 1 or mean.shape[-1] == 0:
        raise ValueError("mean must have a nonempty final action dimension")
    try:
        log_std = torch.broadcast_to(log_std, mean.shape)
    except RuntimeError as error:
        raise ValueError("log_std must broadcast to mean.shape") from error
    return Independent(Normal(mean, log_std.exp()), 1)


def gaussian_action(
    mean: Tensor,
    log_std: Tensor,
    *,
    actions: Tensor | None = None,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(action, summed_log_prob, summed_entropy)`` for a diagonal Gaussian.

    Actions have shape ``mean.shape``; both statistics have shape
    ``mean.shape[:-1]``. Sampling is non-reparameterized, as expected by PPO.
    """
    distribution = diagonal_gaussian(mean, log_std)
    if actions is None:
        actions = mean if deterministic else distribution.sample()
    elif actions.shape != mean.shape:
        raise ValueError("actions must have shape mean.shape")
    return actions, distribution.log_prob(actions), distribution.entropy()
