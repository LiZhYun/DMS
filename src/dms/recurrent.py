"""Optional causal encoding of one observer's local observation history.

The bounded policy path is ``encode_windows([batch, history, observation],
valid)``. Each window starts from zero hidden state and ends at the current
decision. Only left padding is allowed; an all-empty history produces zeros.
Packed GRU execution avoids a Python recurrent loop for PPO minibatches.

Keep raw histories between rollouts. During collection, normalize the complete
window using that rollout's frozen observation moments and store the normalized
window for PPO. Recompute its encoding with current parameters during PPO;
do not cache learned encodings or carry an old parameter-dependent hidden state.
Current intention features and action masks belong downstream of this module.

``forward`` also supports explicit time-major sequences and reset boundaries.
It is a reference/stateful interface, not required by the bounded-window runner.
PPO chunks using it must carry detached initial hidden snapshots and preserve
time ordering. The module never stores mutable recurrent state of its own.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


def _mask(mask: Tensor, shape: tuple[int, ...], device: torch.device, name: str) -> None:
    if mask.dtype != torch.bool:
        raise TypeError(f'{name} must be boolean')
    if tuple(mask.shape) != shape or mask.device != device:
        raise ValueError(f'{name} must have shape {shape} on the observation device')


def append_observation(
    history: Tensor, valid: Tensor, observation: Tensor,
    reset_before: Tensor, active: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return a new time-major history after one local decision observation.

    ``history`` is [history, batch, observation]; ``valid`` is [history, batch].
    ``observation`` is [batch, observation]. Boolean ``reset_before`` and
    ``active`` are [batch]. Reset clears that stream before appending its new
    observation. Inactive/dead streams are cleared and append no observation.
    A revived stream therefore cannot inherit its pre-death history. Call this
    once per decision observation, including the initial reset observation.

    Inputs are not mutated or detached. Collection should call under no_grad;
    clone returned windows when storing them if downstream code mutates them.
    """
    if history.ndim != 3 or history.shape[0] < 1:
        raise ValueError('history must have nonempty shape [history, batch, observation]')
    length, batch, width = history.shape
    if (observation.shape != (batch, width) or observation.device != history.device
            or observation.dtype != history.dtype or not history.is_floating_point()):
        raise ValueError('observation must match history batch, width, floating dtype, and device')
    _mask(valid, (length, batch), history.device, 'valid')
    _mask(reset_before, (batch,), history.device, 'reset_before')
    _mask(active, (batch,), history.device, 'active')
    keep = ~(reset_before | ~active)
    previous_valid = valid[1:] & keep[None]
    previous = torch.where(previous_valid[..., None], history[1:], 0)
    current = torch.where(active[:, None], observation, 0)
    return (torch.cat((previous, current[None]), dim=0),
            torch.cat((previous_valid, active[None]), dim=0))


class LocalHistoryEncoder(nn.Module):
    """One deterministic, unidirectional GRU for one observer's local inputs.

    Batch entries are independent streams; there is no peer or global-state
    input. Actor and critic may instantiate separate encoders, keeping optimizer
    ownership and gradients separate. No dropout or running statistics change
    collection/replay behavior. Optional integration should create this module
    only when enabled, preserving old feed-forward checkpoint keys by default.
    """

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        if any(not isinstance(size, int) or isinstance(size, bool) or size < 1
               for size in (obs_dim, hidden_dim)):
            raise ValueError('obs_dim and hidden_dim must be positive integers')
        self.obs_dim, self.hidden_dim = obs_dim, hidden_dim
        self.gru = nn.GRU(obs_dim, hidden_dim)
        for name, parameter in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)

    def _observations(self, observations: Tensor) -> None:
        if (observations.ndim != 3 or observations.shape[-1] != self.obs_dim
                or not observations.is_floating_point()):
            raise ValueError('observations must have three axes and the configured floating observation width')

    def encode_windows(self, observations: Tensor, valid: Tensor) -> Tensor:
        """Encode [batch, history, observation] windows into [batch, hidden].

        ``valid`` is boolean [batch, history]: false left padding followed by
        current-episode observations. Internal gaps or trailing padding are
        rejected because the final element must represent this decision.
        Empty histories are valid and yield exactly zero, even with learned
        biases. Padding values do not affect outputs or gradients.
        """
        self._observations(observations)
        batch, length, _ = observations.shape
        if length < 1:
            raise ValueError('history must contain at least one slot')
        _mask(valid, (batch, length), observations.device, 'valid')
        if (valid[:, :-1] & ~valid[:, 1:]).any().item():
            raise ValueError('valid must contain only left padding followed by valid observations')
        lengths = valid.sum(dim=1)
        nonempty = lengths.nonzero(as_tuple=True)[0]
        result = observations.new_zeros(batch, self.hidden_dim)
        if nonempty.numel() == 0:
            return result
        # Move each suffix to the front for packing; never expose padding data
        # (including NaNs) to the GRU. pack_padded_sequence ignores right padding.
        clean = torch.where(valid[..., None], observations, 0)
        positions = torch.arange(length, device=observations.device)[None]
        indices = (positions + (length - lengths)[:, None]).clamp_max(length - 1)
        aligned = clean.gather(1, indices[..., None].expand(-1, -1, self.obs_dim))
        packed = pack_padded_sequence(
            aligned[nonempty], lengths[nonempty].cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return result.index_copy(0, nonempty, hidden[0])

    def forward(
        self, observations: Tensor, valid: Tensor,
        reset_before: Tensor | None = None, initial_hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode [time, batch, observation], returning features and final state.

        Features have shape [time, batch, hidden], final state [batch, hidden].
        ``reset_before[t,b]`` clears state before consuming that observation.
        Invalid observations clear state and produce zero; they represent
        padding or inactive/dead streams, never a hidden carry across a gap.

        With no initial state, recurrence starts from zeros. An explicit state
        is used without an implicit detach: callers must pass ``state.detach()``
        when starting a truncated PPO chunk. Resets cut gradients themselves.
        """
        self._observations(observations)
        time, batch, _ = observations.shape
        if time < 1:
            raise ValueError('sequence must contain at least one timestep')
        _mask(valid, (time, batch), observations.device, 'valid')
        if reset_before is None:
            reset_before = torch.zeros_like(valid)
        _mask(reset_before, (time, batch), observations.device, 'reset_before')
        hidden = observations.new_zeros(batch, self.hidden_dim) if initial_hidden is None else initial_hidden
        if (hidden.shape != (batch, self.hidden_dim) or hidden.device != observations.device
                or hidden.dtype != observations.dtype):
            raise ValueError('initial_hidden must match batch, hidden width, observation dtype, and device')
        outputs = []
        for step in range(time):
            keep = valid[step] & ~reset_before[step]
            hidden = torch.where(keep[:, None], hidden, 0)
            current = torch.where(valid[step, :, None], observations[step], 0)
            _, state = self.gru(current[None], hidden[None])
            hidden = torch.where(valid[step, :, None], state[0], 0)
            outputs.append(hidden)
        return torch.stack(outputs), hidden
