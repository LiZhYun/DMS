import pytest
import torch

from dms.sequences import pack_recurrent_chunks


def test_sentinel_packing_preserves_time_environment_agent_and_hidden_state():
    # Different decimal places identify each axis, exposing accidental reshapes.
    times = torch.arange(5)[:, None, None]
    environments = torch.arange(2)[None, :, None]
    agents = torch.arange(3)[None, None, :]
    observations = 100 * times + 10 * environments + agents
    states = observations * 10 + 7
    previous_actions = observations + 10000
    reset_mask = torch.ones(5, 2, dtype=torch.bool)
    reset_mask[2, 1] = False
    result = pack_recurrent_chunks(
        {"observations": observations, "previous_actions": previous_actions, "reset_mask": reset_mask},
        states,
        chunk_length=3,
    )

    torch.testing.assert_close(result.starts, torch.tensor([[0, 0], [0, 1], [3, 0], [3, 1]]))
    assert result.data["observations"].shape == (3, 4, 3)
    torch.testing.assert_close(result.valid, torch.tensor([[True] * 4, [True] * 4, [True, True, False, False]]))
    for chunk, (start, environment) in enumerate(result.starts.tolist()):
        torch.testing.assert_close(result.initial_states[chunk], states[start, environment])
        for offset in range(3):
            if start + offset < 5:
                torch.testing.assert_close(result.data["observations"][offset, chunk], observations[start + offset, environment])
                torch.testing.assert_close(result.data["previous_actions"][offset, chunk], previous_actions[start + offset, environment])
                assert result.data["reset_mask"][offset, chunk] == reset_mask[start + offset, environment]
            else:
                assert torch.count_nonzero(result.data["observations"][offset, chunk]) == 0
                assert not result.data["reset_mask"][offset, chunk]


def test_chunk_minibatch_selection_keeps_each_sequence_contiguous():
    source = torch.arange(24).reshape(6, 2, 2)
    chunks = pack_recurrent_chunks({"x": source}, source + 100, chunk_length=2)
    selected = torch.tensor([5, 0, 3])
    minibatch = chunks.data["x"][:, selected]
    for index, chunk in enumerate(selected.tolist()):
        start, environment = chunks.starts[chunk].tolist()
        torch.testing.assert_close(minibatch[:, index], source[start : start + 2, environment])


def test_chunk_longer_than_rollout_has_valid_padding_mask():
    source = torch.tensor([[1.0], [2.0]])
    result = pack_recurrent_chunks({"x": source}, source, chunk_length=4)
    torch.testing.assert_close(result.data["x"], torch.tensor([[1.0], [2.0], [0.0], [0.0]]))
    torch.testing.assert_close(result.valid, torch.tensor([[True], [True], [False], [False]]))


def test_rejects_mismatched_time_environment_axes():
    with pytest.raises(ValueError, match="prefix"):
        pack_recurrent_chunks({"observations": torch.zeros(3, 2, 4)}, torch.zeros(2, 3, 4), 2)
