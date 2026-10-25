"""Pack contiguous recurrent sequences without mixing environments or timesteps."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RecurrentChunks:
    """Time-major chunks and the recurrent state immediately before each chunk.

    ``data[name]`` is ``[chunk_length, chunks, ...]``. ``valid`` marks real
    timesteps, excluding zero padding in the final partial chunk. ``starts`` is
    ``[chunks, 2]`` and records each chunk's original ``(time, environment)``.
    Chunks are ordered by start time, then environment.
    """

    data: dict[str, Tensor]
    initial_states: Tensor
    valid: Tensor
    starts: Tensor


def pack_recurrent_chunks(
    data: dict[str, Tensor],
    recurrent_states: Tensor,
    chunk_length: int,
) -> RecurrentChunks:
    """Pack tensors with a shared ``[time, environment]`` prefix.

    ``recurrent_states[t, e]`` is the hidden state *before* processing input
    ``data[t, e]``. Any agent or layer axes belong after the first two axes.
    Include episode-start masks in ``data`` so the recurrent model can reset
    within a chunk. Shuffle or minibatch on the chunk axis, never the time axis.
    The caller must exclude ``~valid`` from every loss and normalization.
    """
    if not isinstance(chunk_length, int) or isinstance(chunk_length, bool) or chunk_length < 1:
        raise ValueError("chunk_length must be a positive integer")
    if not data:
        raise ValueError("data must contain at least one tensor")
    if recurrent_states.ndim < 2:
        raise ValueError("recurrent_states must have a [time, environment] prefix")
    time_steps, environments = recurrent_states.shape[:2]
    if time_steps == 0 or environments == 0:
        raise ValueError("time and environment dimensions must be nonempty")
    for name, tensor in data.items():
        if tensor.ndim < 2 or tensor.shape[:2] != (time_steps, environments):
            raise ValueError(f"{name} must have prefix {(time_steps, environments)}")
        if tensor.device != recurrent_states.device:
            raise ValueError(f"{name} must be on the same device as recurrent_states")

    device = recurrent_states.device
    chunk_times = torch.arange(0, time_steps, chunk_length, device=device)
    starts = torch.stack(
        (
            chunk_times.repeat_interleave(environments),
            torch.arange(environments, device=device).repeat(len(chunk_times)),
        ),
        dim=-1,
    )
    times = starts[:, 0].unsqueeze(0) + torch.arange(chunk_length, device=device).unsqueeze(1)
    valid = times < time_steps
    safe_times = times.clamp_max(time_steps - 1)
    packed = {}
    for name, tensor in data.items():
        selected = tensor[safe_times, starts[:, 1]]
        valid_expanded = valid.reshape(*valid.shape, *((1,) * (tensor.ndim - 2)))
        packed[name] = torch.where(valid_expanded, selected, torch.zeros_like(selected))
    return RecurrentChunks(
        data=packed,
        initial_states=recurrent_states[starts[:, 0], starts[:, 1]],
        valid=valid,
        starts=starts,
    )
