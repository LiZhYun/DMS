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
from .recurrent import LocalHistoryEncoder


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
                 coordination: str = 'soft', observer_id: int = 0, *,
                 policy_history_length: int = 0, initial_log_std: float = -0.5):
        super().__init__()
        if coordination not in {'soft', 'causal', 'all'}:
            raise ValueError('coordination must be soft, causal, or all')
        if (not isinstance(policy_history_length, int) or isinstance(policy_history_length, bool)
                or policy_history_length < 0):
            raise ValueError('policy_history_length must be a nonnegative integer')
        if (not isinstance(initial_log_std, (int, float)) or isinstance(initial_log_std, bool)
                or not math.isfinite(initial_log_std) or not -5 <= initial_log_std <= 2):
            raise ValueError('initial_log_std must be a finite number in [-5, 2]')
        self.discrete = discrete
        self.latent_dim = latent_dim
        self.coordination = coordination
        self.observer_id = observer_id
        self.obs_dim = obs_dim
        self.policy_history_length = policy_history_length
        self.obs_moments = RunningMoments(obs_dim)
        self.value_moments = RunningMoments(1)
        if policy_history_length:
            self.actor_encoder = LocalHistoryEncoder(obs_dim, hidden_dim)
            self.critic_encoder = LocalHistoryEncoder(obs_dim, hidden_dim)
        local_width = hidden_dim if policy_history_length else obs_dim
        inputs = local_width + (latent_dim if coordination == 'soft' else num_peers * latent_dim)
        self.actor = mlp(inputs, hidden_dim, action_dim, 0.01)
        self.critic = mlp(inputs, hidden_dim, 1, 1.0)
        # Initialization only: checkpoints restore learned scales, whose runtime
        # clamp remains [-5, 2]. Discrete policies create no scale parameters.
        self.log_std = None if discrete else nn.Parameter(torch.full((action_dim,), float(initial_log_std)))
        if latent_dim and coordination == 'soft':
            self.query = nn.Linear(latent_dim, latent_dim, bias=False)
            self.key = nn.Linear(latent_dim, latent_dim, bias=False)

    def _history_parts(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        """Split local windows into base channels and exact, unnormalized validity."""
        if (observations.ndim < 3 or observations.shape[-2:] !=
                (self.policy_history_length, self.obs_dim + 1)):
            raise ValueError('history observations must end in [policy_history_length, obs_dim + 1]')
        valid = observations[..., -1]
        if not ((valid == 0) | (valid == 1)).all().item():
            raise ValueError('history validity channel must contain exactly 0 or 1')
        return observations[..., :-1], valid.bool()

    def normalize_obs(self, observations: Tensor) -> Tensor:
        base = self._history_parts(observations)[0] if self.policy_history_length else observations
        normalized = ((base - self.obs_moments.mean) / self.obs_moments.std).clamp(-10, 10)
        return (torch.cat((normalized, observations[..., -1:]), dim=-1)
                if self.policy_history_length else normalized)

    @torch.no_grad()
    def update_observation_scale(self, raw_observations: Tensor) -> None:
        """Count each newest live observation once, excluding old/padded windows."""
        if self.policy_history_length:
            base, valid = self._history_parts(raw_observations)
            raw_observations = base[..., -1, :][valid[..., -1]]
        self.obs_moments.update(raw_observations)

    def critic_parameters(self):
        """Parameters for the critic learning-rate group, including its encoder."""
        yield from self.critic.parameters()
        if self.policy_history_length:
            yield from self.critic_encoder.parameters()

    def _local_features(self, observations: Tensor, *, critic: bool = False) -> Tensor:
        if not self.policy_history_length:
            return observations
        base, valid = self._history_parts(observations)
        encoder = self.critic_encoder if critic else self.actor_encoder
        return encoder.encode_windows(base, valid)

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
        local = self._local_features(observations)
        return self._distribution_from_local(local, intentions, relevant, action_mask)

    def _distribution_from_local(self, local: Tensor, intentions: Tensor,
                                 relevant: Tensor, action_mask: Tensor | None = None):
        output = self.actor(self._inputs(local, intentions, relevant))
        return (masked_categorical(output, action_mask) if self.discrete
                else diagonal_gaussian(output, self.log_std.clamp(-5, 2)))

    def normalized_value(self, observations: Tensor, intentions: Tensor, relevant: Tensor) -> Tensor:
        local = self._local_features(observations, critic=True)
        return self.critic(self._inputs(local, intentions, relevant)).squeeze(-1)

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
        # All ablations share the same local history; only peer context changes.
        local = self._local_features(observations)
        full = self._distribution_from_local(local, intentions, visible, action_mask)
        selected = torch.zeros_like(visible)
        for peer in range(visible.shape[1]):
            removed = visible.clone()
            removed[:, peer] = False
            ablated = self._distribution_from_local(local, intentions, removed, action_mask)
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
