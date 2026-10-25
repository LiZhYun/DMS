"""Independent actor/critic with optional inferred intentions.

All tensors describe ONE observer. No centralized state enters this module.
The runner caches normalized observations and inferred intentions for PPO.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.distributions import kl_divergence

from .distributions import masked_categorical, diagonal_gaussian


def mlp(inputs: int, hidden: int, outputs: int, output_gain: float) -> nn.Sequential:
    network = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
                            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, outputs))
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(network[-1].weight, output_gain)
    return network


class RunningMoments(nn.Module):
    """Parallel Welford moments, saved with the policy checkpoint."""
    def __init__(self, size: int):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('variance', torch.ones(size))
        self.register_buffer('count', torch.tensor(1e-4, dtype=torch.float64))

    @property
    def std(self) -> Tensor:
        return self.variance.clamp_min(1e-4).sqrt()

    @torch.no_grad()
    def update(self, values: Tensor) -> None:
        values = values.reshape(-1, self.mean.numel())
        if not len(values):
            return
        n = len(values)
        delta = values.mean(0) - self.mean
        total = self.count + n
        m2 = self.variance * self.count + values.var(0, unbiased=False) * n
        m2 = m2 + delta.square() * self.count * n / total
        self.mean.add_(delta * (n / total))
        self.variance.copy_(m2 / total)
        self.count.copy_(total)


class IndependentAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, num_peers: int,
                 latent_dim: int = 0, hidden_dim: int = 64, discrete: bool = False,
                 coordination: str = 'soft', observer_id: int = 0):
        super().__init__()
        if coordination not in {'soft', 'causal', 'all'}:
            raise ValueError('coordination must be soft, causal, or all')
        self.discrete = discrete
        self.latent_dim = latent_dim
        self.coordination = coordination
        self.observer_id = observer_id
        self.obs_moments = RunningMoments(obs_dim)
        self.value_moments = RunningMoments(1)
        inputs = obs_dim + (latent_dim if coordination == 'soft' else num_peers * latent_dim)
        self.actor = mlp(inputs, hidden_dim, action_dim, 0.01)
        self.critic = mlp(inputs, hidden_dim, 1, 1.0)
        self.log_std = None if discrete else nn.Parameter(torch.full((action_dim,), -0.5))
        if latent_dim and coordination == 'soft':
            self.query = nn.Linear(latent_dim, latent_dim, bias=False)
            self.key = nn.Linear(latent_dim, latent_dim, bias=False)

    def normalize_obs(self, observations: Tensor) -> Tensor:
        return ((observations - self.obs_moments.mean) / self.obs_moments.std).clamp(-10, 10)

    def _inputs(self, observations: Tensor, intentions: Tensor, relevant: Tensor) -> Tensor:
        if not self.latent_dim:
            return observations
        weights = relevant.to(intentions.dtype)
        if self.coordination == 'soft':
            query = self.query(intentions[:, self.observer_id])
            scores = (self.key(intentions) * query[:, None]).sum(-1) / math.sqrt(self.latent_dim)
            scores = scores.masked_fill(~relevant, -torch.inf)
            # An observer with no visible peers gets zero coordination input.
            scores = torch.where(relevant.any(-1, keepdim=True), scores, torch.zeros_like(scores))
            weights = scores.softmax(-1) * relevant
        weighted = intentions * weights[..., None]
        context = weighted.sum(1) if self.coordination == 'soft' else weighted.flatten(1)
        return torch.cat((observations, context), -1)

    def distribution(self, observations: Tensor, intentions: Tensor,
                     relevant: Tensor, action_mask: Tensor | None = None):
        output = self.actor(self._inputs(observations, intentions, relevant))
        return (masked_categorical(output, action_mask) if self.discrete
                else diagonal_gaussian(output, self.log_std.clamp(-5, 2)))

    def normalized_value(self, observations: Tensor, intentions: Tensor, relevant: Tensor) -> Tensor:
        return self.critic(self._inputs(observations, intentions, relevant)).squeeze(-1)

    def value(self, observations: Tensor, intentions: Tensor, relevant: Tensor) -> Tensor:
        return self.normalized_value(observations, intentions, relevant) * self.value_moments.std[0] + self.value_moments.mean[0]

    def act(self, observations: Tensor, intentions: Tensor, relevant: Tensor,
            action_mask: Tensor | None = None, deterministic: bool = False):
        dist = self.distribution(observations, intentions, relevant, action_mask)
        action = (dist.probs.argmax(-1) if self.discrete else dist.mean) if deterministic else dist.sample()
        return action, dist.log_prob(action), self.value(observations, intentions, relevant)

    @torch.no_grad()
    def relevance(self, observations: Tensor, intentions: Tensor, visible: Tensor,
                  action_mask: Tensor | None = None, threshold: float = 0.6) -> Tensor:
        """KL(full policy || one-peer-removed policy), equation (coordination).

        Computed once during collection, then cached during PPO. Thus the
        update evaluates precisely the information/gates used to collect data.
        """
        visible = visible.clone()
        visible[:, self.observer_id] = False
        if not self.latent_dim or self.coordination != 'causal':
            return visible
        full = self.distribution(observations, intentions, visible, action_mask)
        selected = torch.zeros_like(visible)
        for peer in range(visible.shape[1]):
            removed = visible.clone()
            removed[:, peer] = False
            ablated = self.distribution(observations, intentions, removed, action_mask)
            selected[:, peer] = visible[:, peer] & (kl_divergence(full, ablated) > threshold)
        selected[:, self.observer_id] = False
        return selected

    @torch.no_grad()
    def update_value_scale(self, targets: Tensor) -> None:
        """PopArt: normalize value targets while preserving raw predictions."""
        old_mean, old_std = self.value_moments.mean.clone(), self.value_moments.std.clone()
        self.value_moments.update(targets)
        new_mean, new_std = self.value_moments.mean, self.value_moments.std
        head = self.critic[-1]
        head.weight.mul_((old_std / new_std)[:, None])
        head.bias.copy_((old_std * head.bias + old_mean - new_mean) / new_std)
