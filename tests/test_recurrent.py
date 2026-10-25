import copy
import io

import pytest
import torch
from torch import nn

from dms.distributions import masked_categorical
from dms.recurrent import LocalHistoryEncoder, append_observation


def test_packed_windows_match_individually_unrolled_valid_suffixes():
    torch.manual_seed(10)
    encoder = LocalHistoryEncoder(3, 5)
    windows = torch.randn(5, 6, 3)
    lengths = torch.tensor([6, 1, 0, 4, 2])  # deliberately unsorted
    valid = torch.arange(6)[None] >= 6-lengths[:, None]
    windows[~valid] = float('nan')
    # Learned nonzero biases must not turn empty histories into features.
    with torch.no_grad():
        encoder.gru.bias_ih_l0.fill_(.2)
    encoded = encoder.encode_windows(windows, valid)
    for stream, length in enumerate(lengths):
        if length:
            sequence = windows[stream, -length:, None]
            _, state = encoder(sequence, torch.ones(len(sequence), 1, dtype=torch.bool))
            torch.testing.assert_close(encoded[stream], state[0])
        else:
            assert torch.equal(encoded[stream], torch.zeros(5))
    assert torch.isfinite(encoded).all()


def test_reset_and_death_clear_only_the_affected_observer_history():
    history = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
    valid = torch.ones(3, 2, dtype=torch.bool)
    original = history.clone()
    observation = torch.tensor([[100., 101., 102.], [200., 201., 202.]])
    changed, seen = append_observation(history, valid, observation,
                                       torch.tensor([True, False]), torch.tensor([True, True]))
    torch.testing.assert_close(history, original)
    assert not seen[:-1, 0].any() and seen[-1, 0]
    assert torch.equal(changed[:-1, 0], torch.zeros(2, 3))
    torch.testing.assert_close(changed[:-1, 1], history[1:, 1])
    dead, seen = append_observation(changed, seen, observation,
                                   torch.zeros(2, dtype=torch.bool), torch.tensor([True, False]))
    assert not seen[:, 1].any() and torch.equal(dead[:, 1], torch.zeros(3, 3))
    revived, seen = append_observation(dead, seen, observation,
                                      torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    assert not seen[:-1, 1].any() and seen[-1, 1]
    torch.testing.assert_close(revived[-1], observation)


def test_bounded_window_collection_and_shuffled_ppo_replay_have_equal_masked_likelihoods():
    torch.manual_seed(11)
    encoder, head = LocalHistoryEncoder(3, 7), nn.Linear(7, 4)
    history, valid = torch.zeros(4, 2, 3), torch.zeros(4, 2, dtype=torch.bool)
    windows, validity, masks, actions, likelihoods = [], [], [], [], []
    with torch.no_grad():
        for step in range(8):
            reset = torch.tensor([step in (0, 5), step == 0])
            active = torch.tensor([True, step != 3])
            history, valid = append_observation(history, valid, torch.randn(2, 3), reset, active)
            window, seen = history.transpose(0, 1).clone(), valid.T.clone()
            mask = torch.tensor([[True, False, True, True], [True, True, False, True]])
            mask[~active] = torch.tensor([True, False, False, False])
            dist = masked_categorical(head(encoder.encode_windows(window, seen)), mask)
            action = dist.sample()
            windows.append(window); validity.append(seen); masks.append(mask)
            actions.append(action); likelihoods.append(dist.log_prob(action))
    order = torch.randperm(16)
    replay = masked_categorical(head(encoder.encode_windows(torch.cat(windows)[order],
                               torch.cat(validity)[order])), torch.cat(masks)[order])
    actual = replay.log_prob(torch.cat(actions)[order])
    expected = torch.cat(likelihoods)[order]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close((actual-expected).exp(), torch.ones(16), atol=2e-6, rtol=1e-6)
    assert torch.equal(replay.probs[~torch.cat(masks)[order]], torch.zeros_like(replay.probs[~torch.cat(masks)[order]]))


def test_time_major_replay_matches_collection_and_initial_hidden_snapshot():
    torch.manual_seed(12)
    encoder = LocalHistoryEncoder(2, 4)
    observations = torch.randn(7, 3, 2)
    valid = torch.ones(7, 3, dtype=torch.bool)
    valid[3, 1] = False
    reset = torch.zeros_like(valid); reset[4, 0] = True
    initial = torch.randn(3, 4)
    state, collected = initial.clone(), []
    for step in range(7):
        feature, state = encoder(observations[step:step+1], valid[step:step+1],
                                 reset[step:step+1], state)
        collected.append(feature)
    replay, final = encoder(observations, valid, reset, initial)
    torch.testing.assert_close(replay, torch.cat(collected), atol=0, rtol=0)
    torch.testing.assert_close(final, state, atol=0, rtol=0)
    _, boundary = encoder(observations[:3], valid[:3], reset[:3], initial)
    suffix, _ = encoder(observations[3:], valid[3:], reset[3:], boundary.detach())
    torch.testing.assert_close(suffix, replay[3:], atol=0, rtol=0)


def test_no_future_or_other_stream_can_change_an_earlier_local_feature():
    torch.manual_seed(13)
    encoder = LocalHistoryEncoder(2, 4)
    observations = torch.randn(6, 2, 2)
    valid = torch.ones(6, 2, dtype=torch.bool)
    before, _ = encoder(observations, valid)
    changed = observations.clone(); changed[3:, 0] += 100; changed[:, 1] -= 100
    after, _ = encoder(changed, valid)
    torch.testing.assert_close(after[:3, 0], before[:3, 0], atol=0, rtol=0)
    encoded = encoder.encode_windows(observations.transpose(0, 1), valid.T)
    altered = observations.clone(); altered[:, 1] += 100
    encoded_altered = encoder.encode_windows(altered.transpose(0, 1), valid.T)
    torch.testing.assert_close(encoded_altered[0], encoded[0], atol=0, rtol=0)


def test_reset_and_detached_chunk_boundaries_cut_gradients():
    torch.manual_seed(14)
    encoder = LocalHistoryEncoder(2, 4)
    observations = torch.randn(6, 1, 2, requires_grad=True)
    valid = torch.ones(6, 1, dtype=torch.bool)
    reset = torch.zeros_like(valid); reset[3] = True
    _, final = encoder(observations, valid, reset)
    final.square().sum().backward()
    assert torch.equal(observations.grad[:3], torch.zeros_like(observations.grad[:3]))
    assert observations.grad[3:].abs().sum() > 0
    encoder.zero_grad(set_to_none=True)
    observations.grad = None
    _, state = encoder(observations[:3], valid[:3])
    _, final = encoder(observations[3:], valid[3:], initial_hidden=state.detach())
    final.square().sum().backward()
    assert torch.equal(observations.grad[:3], torch.zeros_like(observations.grad[:3]))
    assert observations.grad[3:].abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in encoder.parameters())


def test_invalid_death_step_discards_state_and_predeath_gradients():
    torch.manual_seed(17)
    encoder = LocalHistoryEncoder(2, 4)
    observations = torch.randn(6, 2, 2, requires_grad=True)
    valid = torch.ones(6, 2, dtype=torch.bool); valid[3, 0] = False
    features, final = encoder(observations, valid, initial_hidden=torch.ones(2, 4))
    assert torch.equal(features[3, 0], torch.zeros(4))
    _, restarted = encoder(observations[4:, :1], valid[4:, :1])
    torch.testing.assert_close(final[0], restarted[0])
    final[0].square().sum().backward()
    assert torch.equal(observations.grad[:4, 0], torch.zeros_like(observations.grad[:4, 0]))
    assert observations.grad[4:, 0].abs().sum() > 0
    assert torch.equal(observations.grad[:, 1], torch.zeros_like(observations.grad[:, 1]))


def test_window_gradients_ignore_padding_and_recompute_after_parameter_update():
    torch.manual_seed(15)
    encoder = LocalHistoryEncoder(2, 4)
    observations = torch.randn(2, 5, 2, requires_grad=True)
    valid = torch.tensor([[False, False, True, True, True], [False]*5])
    before = encoder.encode_windows(observations, valid)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=.2)
    before[0].square().sum().backward()
    assert torch.equal(observations.grad[~valid], torch.zeros_like(observations.grad[~valid]))
    assert observations.grad[valid].abs().sum() > 0
    optimizer.step()
    after = encoder.encode_windows(observations, valid)
    assert not torch.allclose(before[0], after[0])
    assert torch.equal(after[1], torch.zeros(4))


def test_checkpoint_roundtrip_and_evaluation_leave_parameters_and_buffers_unchanged():
    torch.manual_seed(16)
    encoder = LocalHistoryEncoder(2, 4)
    windows = torch.randn(3, 5, 2)
    valid = torch.ones(3, 5, dtype=torch.bool)
    before = copy.deepcopy(encoder.state_dict())
    buffer = io.BytesIO(); torch.save(before, buffer); buffer.seek(0)
    restored = LocalHistoryEncoder(2, 4)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    encoder.train(); restored.eval()
    torch.testing.assert_close(encoder.encode_windows(windows, valid),
                               restored.encode_windows(windows, valid), atol=0, rtol=0)
    for name, value in encoder.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert not list(encoder.buffers())


@pytest.mark.parametrize('valid', [torch.tensor([[True, False, True]]),
                                  torch.tensor([[True, True, False]])])
def test_window_rejects_internal_gaps_or_trailing_padding(valid):
    with pytest.raises(ValueError, match='left padding'):
        LocalHistoryEncoder(2, 4).encode_windows(torch.randn(1, 3, 2), valid)


def test_all_empty_windows_are_zero_and_shape_or_mask_errors_are_explicit():
    encoder = LocalHistoryEncoder(2, 4)
    empty = torch.full((2, 3, 2), float('nan'))
    assert torch.equal(encoder.encode_windows(empty, torch.zeros(2, 3, dtype=torch.bool)), torch.zeros(2, 4))
    with pytest.raises(TypeError, match='boolean'):
        encoder.encode_windows(empty, torch.zeros(2, 3))
    with pytest.raises(ValueError, match='shape'):
        encoder.encode_windows(empty, torch.zeros(3, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match='at least one slot'):
        encoder.encode_windows(empty[:, :0], torch.zeros(2, 0, dtype=torch.bool))
