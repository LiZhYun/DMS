"""Per-observer action-only CVAE used by Deep Motor System.

History tensors are [batch, time, peer, padded action feature]; visibility is
[batch, time, peer]. Discrete actions are one-hot vectors. A runner constructs
one independent model for each observer and supplies only actions that observer
could see. There is deliberately no state, observation, or reward argument.

The same GRU, layer normalization, and Gaussian head encode the prior's past
and the posterior's completed past+future sequence. Only auxiliary training
calls ``posterior``. Policies use cached deterministic ``features`` (prior means).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DiagonalGaussian:
    mean: Tensor
    log_std: Tensor

    def rsample(self) -> Tensor:
        return self.mean + self.log_std.exp() * torch.randn_like(self.mean)


def gaussian_kl(q: DiagonalGaussian, p: DiagonalGaussian) -> Tensor:
    """Analytic KL(q || p), summed over the final (latent/event) dimension.

    The manuscript prints the logarithmic ratio with its sign reversed. The
    correct term is log(sigma_p / sigma_q), as implemented here.
    """
    return (
        p.log_std - q.log_std
        + 0.5 * (
            (2 * (q.log_std - p.log_std)).exp()
            + (q.mean - p.mean).square() * (-2 * p.log_std).exp()
            - 1
        )
    ).sum(dim=-1)


class IntentionModel(nn.Module):
    """Independent observer CVAE; encoder weights are shared across its peers.

    ``action_dims`` gives the real continuous dimensions or category counts
    per peer; padded dimensions are ignored in both inputs and likelihoods.
    ``observation_mask`` additionally limits continuous action components to
    those this observer can sense. It is static, per-observer, and [peer, dim].
    Categorical actions are hidden as a whole via ``visible``, never by category.
    ``decoder_history=False`` removes the encoded-history input to decoder
    initialization. The latent, last-action anchor, and peer identity remain.
    """

    def __init__(
        self,
        num_peers: int,
        action_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        discrete: bool = False,
        action_dims: Sequence[int] | None = None,
        observation_mask: Tensor | Sequence[Sequence[bool]] | None = None,
        *,
        decoder_history: bool = True,
    ) -> None:
        super().__init__()
        if min(num_peers, action_dim, hidden_dim, latent_dim) < 1:
            raise ValueError("Model dimensions must be positive")
        dims = list(action_dims) if action_dims is not None else [action_dim] * num_peers
        if len(dims) != num_peers or any(d < 1 or d > action_dim for d in dims):
            raise ValueError("action_dims must give one valid dimension per peer")
        self.num_peers = num_peers
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.discrete = discrete
        self.decoder_history = decoder_history
        self.register_buffer("peer_ids", torch.eye(num_peers))
        action_mask = torch.arange(action_dim)[None, :] < torch.tensor(dims)[:, None]
        if observation_mask is not None:
            if discrete:
                raise ValueError("Categorical actions use whole-action visibility, not observation_mask")
            observed = torch.as_tensor(observation_mask, dtype=torch.bool, device=action_mask.device)
            if observed.shape != (num_peers, action_dim):
                raise ValueError("observation_mask must have shape [num_peers,action_dim]")
            action_mask &= observed
        self.register_buffer("action_mask", action_mask)
        self.encoder = nn.GRU(action_dim + num_peers + 1, hidden_dim, batch_first=True)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.gaussian_head = nn.Linear(hidden_dim, 2 * latent_dim)
        decoder_input_dim = hidden_dim + latent_dim if decoder_history else latent_dim
        self.decoder_start = nn.Linear(decoder_input_dim, hidden_dim)
        self.decoder_cell = nn.GRUCell(action_dim + latent_dim + num_peers, hidden_dim)
        self.decoder_head = nn.Linear(hidden_dim, action_dim)
        if not discrete:
            self.decoder_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
            # Initially predict no action change, as motivated by the paper.
            nn.init.zeros_(self.decoder_head.weight)
            nn.init.zeros_(self.decoder_head.bias)

    def _check(self, actions: Tensor, visible: Tensor) -> None:
        if actions.ndim != 4 or actions.shape[2:] != (self.num_peers, self.action_dim):
            raise ValueError("actions must have shape [B,T,num_peers,action_dim]")
        if visible.shape != actions.shape[:-1]:
            raise ValueError("visible must have shape [B,T,num_peers]")

    def _masked_actions(self, actions: Tensor, visible: Tensor) -> Tensor:
        self._check(actions, visible)
        # where, unlike multiplication by zero, also removes invisible NaNs.
        mask = visible.bool().unsqueeze(-1) & self.action_mask
        return torch.where(mask, actions, torch.zeros_like(actions))

    def _last_visible_action(self, past: Tensor, visible: Tensor) -> Tensor:
        """Latest sensed action per component; zero only when none was sensed."""
        clean = self._masked_actions(past, visible)
        if past.shape[1] == 0:
            return past.new_zeros(past.shape[0], self.num_peers, self.action_dim)
        sensed = visible.bool().unsqueeze(-1) & self.action_mask
        times = torch.arange(past.shape[1], device=past.device)[None, :, None, None]
        last = torch.where(sensed, times, -1).amax(dim=1)
        latest = clean.gather(1, last.clamp_min(0).unsqueeze(1)).squeeze(1)
        return torch.where(last >= 0, latest, torch.zeros_like(latest))

    def _reconstruction_visibility(self, visible: Tensor, future_visible: Tensor) -> Tensor:
        """Continuous deltas require two consecutively observed endpoints.

        A missing predecessor must not become an observed zero action. The
        decoder can still forecast across that gap, but the corresponding
        transition is not a supervised reconstruction target.
        """
        if self.discrete:
            return future_visible.bool()
        initial = (
            visible[:, -1:] if visible.shape[1]
            else future_visible.new_zeros(future_visible.shape[0], 1, self.num_peers)
        )
        predecessor = torch.cat((initial, future_visible[:, :-1]), dim=1)
        return future_visible.bool() & predecessor.bool()

    def _encode(self, actions: Tensor, visible: Tensor) -> Tensor:
        visible = visible.bool() & self.action_mask.any(dim=-1)
        actions = self._masked_actions(actions, visible)
        batch, steps = actions.shape[:2]
        if steps == 0:
            # An empty history is one unknown action with a known peer ID.
            actions = actions.new_zeros(batch, 1, self.num_peers, self.action_dim)
            visible = visible.new_zeros(batch, 1, self.num_peers)
            steps = 1
        ids = self.peer_ids.expand(batch, steps, -1, -1)
        inputs = torch.cat((actions, visible.to(actions.dtype).unsqueeze(-1), ids), dim=-1)
        # Each peer is a separate sequence, never a temporal or batch position.
        inputs = inputs.permute(0, 2, 1, 3).reshape(batch * self.num_peers, steps, -1)
        _, hidden = self.encoder(inputs)
        return self.encoder_norm(hidden[0]).reshape(batch, self.num_peers, -1)

    def _distribution(self, hidden: Tensor) -> DiagonalGaussian:
        mean, log_std = self.gaussian_head(hidden).chunk(2, dim=-1)
        return DiagonalGaussian(mean, log_std.clamp(-5.0, 2.0))

    def prior_parameters(self):
        """Execution encoder parameters; decoder parameters receive ELBO only."""
        for module in (self.encoder, self.encoder_norm, self.gaussian_head):
            yield from module.parameters()

    def prior(self, past: Tensor, visible: Tensor) -> DiagonalGaussian:
        """Execution path: observed history only, one distribution per peer."""
        return self._distribution(self._encode(past, visible))

    def features(self, past: Tensor, visible: Tensor) -> Tensor:
        """Deterministic, differentiable actor features; rollout callers detach them."""
        return self.prior(past, visible).mean

    def posterior(
        self, past: Tensor, visible: Tensor, future: Tensor, future_visible: Tensor
    ) -> DiagonalGaussian:
        """Training only: encode a completed episode's past and future suffix."""
        self._check(past, visible)
        self._check(future, future_visible)
        return self.prior(
            torch.cat((past, future), dim=1),
            torch.cat((visible, future_visible), dim=1),
        )

    def decode(
        self, past: Tensor, visible: Tensor, latent: Tensor, horizon: int
    ) -> DiagonalGaussian | Tensor:
        """Predict a suffix without receiving its reconstruction targets.

        Continuous predictions are Gaussian action changes added to the previous
        predicted mean (the first offset is the most recent visible past action,
        or zero if no past action was visible). Thus NLL is evaluated against
        actual future actions, with the same
        likelihood as a delta relative to that deterministic offset. Discrete
        predictions are logits. Subsequent decoder inputs use its own predicted
        actions/probabilities, with no teacher forcing or future information.
        When ``decoder_history`` is false, earlier past actions can affect the
        prediction only through the supplied latent and last-action anchor.
        """
        if horizon < 1:
            raise ValueError("horizon must be positive")
        batch = past.shape[0]
        if latent.shape != (batch, self.num_peers, self.latent_dim):
            raise ValueError("latent must have shape [B,num_peers,latent_dim]")
        decoder_input = latent
        if self.decoder_history:
            history = self._encode(past, visible)
            decoder_input = torch.cat((history, latent), dim=-1)
        hidden = torch.tanh(self.decoder_start(decoder_input))
        hidden = hidden.reshape(batch * self.num_peers, -1)
        if self.discrete:
            clean_past = self._masked_actions(past, visible)
            previous = (
                clean_past[:, -1] if past.shape[1]
                else past.new_zeros(batch, self.num_peers, self.action_dim)
            )
        else:
            previous = self._last_visible_action(past, visible)
        ids = self.peer_ids.expand(batch, -1, -1)
        predictions = []
        for _ in range(horizon):
            inputs = torch.cat((previous, latent, ids), dim=-1)
            hidden = self.decoder_cell(inputs.reshape(batch * self.num_peers, -1), hidden)
            output = self.decoder_head(hidden).reshape(batch, self.num_peers, self.action_dim)
            if self.discrete:
                output = output.masked_fill(~self.action_mask, -torch.inf)
                previous = output.softmax(dim=-1)
            else:
                output = (previous + output).masked_fill(~self.action_mask, 0.0)
                previous = output
            predictions.append(output)
        prediction = torch.stack(predictions, dim=1)
        if self.discrete:
            return prediction
        log_std = self.decoder_log_std.clamp(-5.0, 2.0).expand_as(prediction)
        return DiagonalGaussian(prediction, log_std)

    def reconstruction_nll(
        self, prediction: DiagonalGaussian | Tensor, future: Tensor, future_visible: Tensor
    ) -> Tensor:
        """Mean negative log likelihood over observed time/peer actions.

        Continuous event dimensions are summed; padded dimensions are excluded.
        The target is the observed action, never the model's own output/sample.
        ``loss`` supplies a mask requiring both observed delta endpoints for
        continuous targets; this primitive evaluates the supplied valid targets.
        """
        target = self._masked_actions(future, future_visible)
        if self.discrete:
            if not isinstance(prediction, Tensor):
                raise TypeError("Discrete prediction must be logits")
            token_nll = F.cross_entropy(
                prediction.reshape(-1, self.action_dim),
                target.argmax(dim=-1).reshape(-1),
                reduction="none",
            ).reshape(future_visible.shape)
        else:
            if not isinstance(prediction, DiagonalGaussian):
                raise TypeError("Continuous prediction must be a DiagonalGaussian")
            element_nll = (
                0.5 * (target - prediction.mean).square() * (-2 * prediction.log_std).exp()
                + prediction.log_std + 0.5 * torch.log(target.new_tensor(2.0 * torch.pi))
            )
            token_nll = torch.where(self.action_mask, element_nll, 0.0).sum(dim=-1)
        observed = future_visible.bool() & self.action_mask.any(dim=-1)
        return torch.where(observed, token_nll, 0.0).sum() / observed.sum().clamp_min(1)

    def loss(
        self,
        past: Tensor,
        visible: Tensor,
        future: Tensor,
        future_visible: Tensor,
        beta: float = 1.0,
        samples: int = 1,
    ) -> dict[str, Tensor]:
        """Time-normalized variational loss with sample-mean reconstruction.

        NLL averages observed time/peer targets; KL averages peer sequences.
        With H fully observed future steps, beta therefore gives a sequence
        KL weight of beta*H. It is not the canonical sequence ELBO at beta=1
        unless H=1. See docs/INTENTION_DIAGNOSIS.md for the exact masked ratio.
        """
        if samples < 1 or future.shape[1] < 1:
            raise ValueError("samples and future horizon must be positive")
        q = self.posterior(past, visible, future, future_visible)
        p = self.prior(past, visible)
        target_visible = self._reconstruction_visibility(visible, future_visible)
        nll = torch.stack([
            self.reconstruction_nll(
                self.decode(past, visible, q.rsample(), future.shape[1]), future, target_visible
            )
            for _ in range(samples)
        ]).mean()
        has_target = target_visible.any(dim=1) & self.action_mask.any(dim=-1)
        per_peer_kl = gaussian_kl(q, p)
        kl = torch.where(has_target, per_peer_kl, 0.0).sum() / has_target.sum().clamp_min(1)
        return {"loss": nll + beta * kl, "nll": nll, "kl": kl}

    @torch.no_grad()
    def filtered_history(
        self, past: Tensor, visible: Tensor, length: int = 4, threshold: float = 1e4
    ) -> tuple[Tensor, Tensor]:
        """Return (filtered visibility [B,T,P], rejected peers [B,P]).

        Compare full-history and recent-history priors, both from this same model.
        The paper does not specify KL direction; we use KL(long || short). If it
        exceeds K, mask the latest l actions for that peer and retain peer IDs.
        The caller passes this mask to ``features``; input tensors are not mutated.
        """
        if length < 1 or threshold < 0:
            raise ValueError("length must be positive and threshold nonnegative")
        long = self.prior(past, visible)
        short = self.prior(past[:, -length:], visible[:, -length:])
        rejected = (gaussian_kl(long, short) > threshold) & visible[:, -length:].bool().any(dim=1)
        filtered = visible.bool().clone()
        filtered[:, -length:] &= ~rejected.unsqueeze(1)
        return filtered, rejected
