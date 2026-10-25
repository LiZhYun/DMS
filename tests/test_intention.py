"""Check information boundaries and the CVAE objective, not training scores."""

import math

import pytest
import torch
from torch.distributions import Independent, Normal, kl_divergence

from dms.intention import DiagonalGaussian, IntentionModel, gaussian_kl


def example(discrete=False):
    torch.manual_seed(31)
    model = IntentionModel(3, 4, hidden_dim=16, latent_dim=5, discrete=discrete)
    actions = torch.randn(2, 7, 3, 4)
    if discrete:
        actions = torch.nn.functional.one_hot(torch.randint(4, (2, 7, 3)), 4).float()
    visible = torch.rand(2, 7, 3) > 0.3
    return model, actions[:, :4], visible[:, :4], actions[:, 4:], visible[:, 4:]


def test_analytic_kl_matches_torch_and_is_zero_for_same_distribution():
    torch.manual_seed(1)
    q = DiagonalGaussian(torch.randn(2, 3, 5), torch.randn(2, 3, 5) * 0.2)
    p = DiagonalGaussian(torch.randn(2, 3, 5), torch.randn(2, 3, 5) * 0.2)
    expected = kl_divergence(
        Independent(Normal(q.mean, q.log_std.exp()), 1),
        Independent(Normal(p.mean, p.log_std.exp()), 1),
    )
    torch.testing.assert_close(gaussian_kl(q, p), expected)
    torch.testing.assert_close(gaussian_kl(q, q), torch.zeros(2, 3), atol=0, rtol=0)


@pytest.mark.parametrize("discrete", [False, True])
def test_unobserved_actions_cannot_change_prior_posterior_or_loss(discrete):
    model, past, visible, future, future_visible = example(discrete)
    altered_past = past.masked_fill(~visible.unsqueeze(-1), float("nan"))
    altered_future = future.masked_fill(~future_visible.unsqueeze(-1), float("nan"))
    torch.testing.assert_close(model.features(past, visible), model.features(altered_past, visible))
    q = model.posterior(past, visible, future, future_visible)
    altered_q = model.posterior(altered_past, visible, altered_future, future_visible)
    torch.testing.assert_close(q.mean, altered_q.mean)
    torch.testing.assert_close(q.log_std, altered_q.log_std)
    torch.manual_seed(22)
    original_loss = model.loss(past, visible, future, future_visible, samples=2)
    torch.manual_seed(22)
    altered_loss = model.loss(altered_past, visible, altered_future, future_visible, samples=2)
    for key in original_loss:
        torch.testing.assert_close(original_loss[key], altered_loss[key])


def test_prior_is_deterministic_and_does_not_take_future():
    model, past, visible, future, future_visible = example()
    original = model.features(past, visible)
    torch.manual_seed(123)
    model.posterior(past, visible, future + 1000, future_visible)
    torch.testing.assert_close(original, model.features(past, visible), atol=0, rtol=0)
    with pytest.raises(TypeError):
        model.prior(past, visible, future)


def test_shared_encoder_and_heads_are_used_for_posterior():
    model, past, visible, future, future_visible = example()
    posterior = model.posterior(past, visible, future, future_visible)
    joined = model.prior(torch.cat((past, future), 1), torch.cat((visible, future_visible), 1))
    torch.testing.assert_close(posterior.mean, joined.mean, atol=0, rtol=0)
    torch.testing.assert_close(posterior.log_std, joined.log_std, atol=0, rtol=0)


def test_peers_and_batch_sequences_are_not_mixed_and_missing_peers_have_ids():
    model, past, visible, _, _ = example()
    all_missing = torch.zeros_like(visible)
    missing_features = model.features(past, all_missing)
    assert not torch.allclose(missing_features[:, 0], missing_features[:, 1])
    original = model.features(past, visible)
    changed = past.clone()
    changed[0, :, 0] += 100
    altered = model.features(changed, visible)
    torch.testing.assert_close(original[1], altered[1])
    torch.testing.assert_close(original[0, 1:], altered[0, 1:])
    assert not torch.allclose(original[0, 0], altered[0, 0])


def test_continuous_nll_uses_actual_targets_and_exact_gaussian_likelihood():
    model = IntentionModel(2, 3, action_dims=[2, 3])
    mean = torch.zeros(1, 2, 2, 3)
    prediction = DiagonalGaussian(mean, torch.zeros_like(mean))
    targets = torch.ones_like(mean)
    visible = torch.ones(1, 2, 2, dtype=torch.bool)
    nll = model.reconstruction_nll(prediction, targets, visible)
    expected = 2.5 * (0.5 + 0.5 * math.log(2 * math.pi))
    assert nll.item() == pytest.approx(expected)
    increased = model.reconstruction_nll(prediction, targets * 3, visible)
    assert increased > nll
    # The padded third coordinate of peer 0 cannot affect the target loss.
    targets[:, :, 0, 2] = float("nan")
    torch.testing.assert_close(nll, model.reconstruction_nll(prediction, targets, visible))


def test_discrete_nll_uses_labels_and_ignores_padding():
    model = IntentionModel(2, 4, discrete=True, action_dims=[2, 4])
    logits = torch.tensor([[[[2.0, 0.0, -torch.inf, -torch.inf], [0.0, 0.0, 0.0, 2.0]]]])
    labels = torch.tensor([[[0, 3]]])
    targets = torch.nn.functional.one_hot(labels, 4).float()
    visible = torch.ones(1, 1, 2, dtype=torch.bool)
    expected = -(logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1))).mean()
    torch.testing.assert_close(model.reconstruction_nll(logits, targets, visible), expected)
    wrong_targets = torch.nn.functional.one_hot(torch.tensor([[[1, 0]]]), 4).float()
    assert model.reconstruction_nll(logits, wrong_targets, visible) > expected


@pytest.mark.parametrize("discrete", [False, True])
def test_elbo_has_finite_gradients_and_all_missing_suffix_has_zero_loss(discrete):
    model, past, visible, future, future_visible = example(discrete)
    terms = model.loss(past, visible, future, future_visible, beta=0.3, samples=3)
    torch.testing.assert_close(terms["loss"], terms["nll"] + 0.3 * terms["kl"])
    terms["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    zero = model.loss(past, visible, future, torch.zeros_like(future_visible))
    assert all(term.item() == 0 for term in zero.values())


@pytest.mark.parametrize("discrete", [False, True])
def test_decoder_never_receives_future_targets_and_samples_are_averaged(discrete):
    model, past, visible, future, future_visible = example(discrete)
    latent = model.posterior(past, visible, future, future_visible).mean
    prediction = model.decode(past, visible, latent, future.shape[1])
    # Hold the posterior draw fixed: likelihood must still depend on data.
    changed_future = future.roll(1, -1) if discrete else future + 10
    nll = model.reconstruction_nll(prediction, future, future_visible)
    changed_nll = model.reconstruction_nll(prediction, changed_future, future_visible)
    assert not torch.isclose(nll, changed_nll)
    q = model.posterior(past, visible, future, future_visible)
    valid_targets = future_visible if discrete else (
        future_visible & torch.cat((visible[:, -1:], future_visible[:, :-1]), dim=1)
    )
    torch.manual_seed(4)
    expected = torch.stack([
        model.reconstruction_nll(model.decode(past, visible, q.rsample(), future.shape[1]), future, valid_targets)
        for _ in range(3)
    ]).mean()
    torch.manual_seed(4)
    actual = model.loss(past, visible, future, future_visible, samples=3)["nll"]
    torch.testing.assert_close(actual, expected)


def test_short_history_filter_masks_only_recent_actions_and_does_not_mutate_input():
    model, past, _, _, _ = example()
    visible = torch.ones(past.shape[:-1], dtype=torch.bool)
    filtered, rejected = model.filtered_history(past, visible, length=2, threshold=0.0)
    assert rejected.all()
    assert filtered[:, :2].all()
    assert not filtered[:, -2:].any()
    assert visible.all()
    retained, rejected = model.filtered_history(past, visible, length=2, threshold=float("inf"))
    assert not rejected.any()
    torch.testing.assert_close(retained, visible)


def test_empty_history_is_valid_and_contains_no_future_information():
    model, past, visible, _, _ = example()
    features = model.features(past[:, :0], visible[:, :0])
    assert features.shape == (2, 3, 5)
    assert torch.isfinite(features).all()


def test_local_continuous_component_mask_blocks_input_and_target_information():
    import numpy as np

    torch.manual_seed(42)
    sensor_mask = np.array([[True, False, True], [False, False, False]], dtype=bool)
    model = IntentionModel(2, 3, observation_mask=sensor_mask, hidden_dim=16)
    actions = torch.randn(2, 7, 2, 3)
    visible = torch.ones(2, 7, 2, dtype=torch.bool)
    changed = actions.masked_fill(~torch.as_tensor(sensor_mask), float("nan"))
    past, future = actions[:, :4], actions[:, 4:]
    changed_past, changed_future = changed[:, :4], changed[:, 4:]
    prior = model.prior(past, visible[:, :4])
    changed_prior = model.prior(changed_past, visible[:, :4])
    torch.testing.assert_close(prior.mean, changed_prior.mean)
    torch.testing.assert_close(prior.log_std, changed_prior.log_std)
    posterior = model.posterior(past, visible[:, :4], future, visible[:, 4:])
    changed_posterior = model.posterior(changed_past, visible[:, :4], changed_future, visible[:, 4:])
    torch.testing.assert_close(posterior.mean, changed_posterior.mean)
    torch.testing.assert_close(posterior.log_std, changed_posterior.log_std)
    torch.manual_seed(91)
    original = model.loss(past, visible[:, :4], future, visible[:, 4:], samples=2)
    torch.manual_seed(91)
    altered = model.loss(changed_past, visible[:, :4], changed_future, visible[:, 4:], samples=2)
    for key in original:
        torch.testing.assert_close(original[key], altered[key])
    prediction = DiagonalGaussian(torch.zeros_like(future), torch.zeros_like(future))
    nll = model.reconstruction_nll(prediction, torch.ones_like(future), visible[:, 4:])
    # Only peer 0's two locally observed dimensions contribute or normalize NLL.
    assert nll.item() == pytest.approx(1 + math.log(2 * math.pi))
    original["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_observation_mask_intersects_padding_and_rejects_categorical_subsets():
    model = IntentionModel(2, 3, action_dims=[1, 3], observation_mask=[[True, True, True], [True, False, True]])
    torch.testing.assert_close(model.action_mask, torch.tensor([[True, False, False], [True, False, True]]))
    with pytest.raises(ValueError, match="whole-action visibility"):
        IntentionModel(2, 3, discrete=True, observation_mask=[[True, False, True]] * 2)
    with pytest.raises(ValueError, match="shape"):
        IntentionModel(2, 3, observation_mask=[True, False, True])


@pytest.mark.parametrize("decoder_history", [False, True])
def test_continuous_decoder_starts_at_last_observed_action_across_missing_suffix(decoder_history):
    model = IntentionModel(3, 3, action_dims=[2, 3, 3],
                           observation_mask=[[True, True, True], [True, False, True], [True] * 3],
                           decoder_history=decoder_history)
    past = torch.tensor([[[[1., 2., 900.], [3., 900., 4.], [900.] * 3],
                          [[5., 6., 900.], [900.] * 3, [900.] * 3],
                          [[900.] * 3, [900.] * 3, [900.] * 3]]])
    visible = torch.tensor([[[True, True, False], [True, False, False], [False] * 3]])
    past = past.masked_fill(~visible.unsqueeze(-1), float("nan"))
    latent = torch.zeros(1, 3, model.latent_dim)
    # Zero-initialized delta head is a persistence forecast. Every peer has a
    # different last observation time; padded and unsensed components stay zero.
    prediction = model.decode(past, visible, latent, horizon=3)
    expected = torch.tensor([[[5., 6., 0.], [3., 0., 4.], [0., 0., 0.]]])
    torch.testing.assert_close(prediction.mean, expected[:, None].expand(-1, 3, -1, -1))
    empty = model.decode(past[:, :0], visible[:, :0], latent, horizon=2)
    torch.testing.assert_close(empty.mean, torch.zeros(1, 2, 3, 3))


@pytest.mark.parametrize("empty_history", [False, True])
def test_continuous_reconstruction_requires_observed_predecessor_and_target(empty_history):
    model = IntentionModel(1, 1, hidden_dim=8, latent_dim=2)
    past = torch.tensor([[[[7.]], [[float("nan")]]]])
    visible = torch.tensor([[[True], [False]]])
    if empty_history:
        past, visible = past[:, :0], visible[:, :0]
    future = torch.tensor([[[[1.]], [[2.]], [[float("nan")]], [[4.]], [[5.]]]])
    future_visible = torch.tensor([[[True], [True], [False], [True], [True]]])
    # First target has no immediate observed predecessor, the hidden third
    # target is excluded, and reappearing fourth target also lacks a predecessor.
    expected_valid = torch.tensor([[[False], [True], [False], [False], [True]]])
    q = model.posterior(past, visible, future, future_visible)
    torch.manual_seed(14)
    prediction = model.decode(past, visible, q.rsample(), horizon=5)
    expected_nll = model.reconstruction_nll(prediction, future, expected_valid)
    torch.manual_seed(14)
    actual = model.loss(past, visible, future, future_visible)
    torch.testing.assert_close(actual["nll"], expected_nll)
    actual["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("discrete", [False, True])
def test_isolated_future_observations_are_targets_only_for_discrete_actions(discrete):
    model = IntentionModel(1, 2, hidden_dim=8, latent_dim=2, discrete=discrete)
    past = torch.zeros(1, 1, 1, 2)
    visible = torch.zeros(1, 1, 1, dtype=torch.bool)
    future = torch.tensor([[[[1., 0.]], [[float("nan"), float("nan")]], [[0., 1.]]]])
    future_visible = torch.tensor([[[True], [False], [True]]])
    terms = model.loss(past, visible, future, future_visible)
    if discrete:
        assert terms["nll"] > 0
    else:
        assert all(term.item() == 0 for term in terms.values())


@pytest.mark.parametrize("discrete", [False, True])
def test_default_decoder_preserves_history_checkpoint_shapes_and_behavior(discrete):
    torch.manual_seed(83)
    original = IntentionModel(2, 3, hidden_dim=8, latent_dim=2, discrete=discrete)
    torch.manual_seed(83)
    explicit = IntentionModel(2, 3, hidden_dim=8, latent_dim=2, discrete=discrete,
                              decoder_history=True)
    assert original.decoder_start.weight.shape == (8, 10)
    assert original.state_dict().keys() == explicit.state_dict().keys()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], atol=0, rtol=0)
    explicit.load_state_dict(original.state_dict(), strict=True)
    past = torch.randn(2, 4, 2, 3)
    visible = torch.ones(2, 4, 2, dtype=torch.bool)
    latent = torch.randn(2, 2, 2)
    before = original.decode(past, visible, latent, horizon=3)
    after = explicit.decode(past, visible, latent, horizon=3)
    if discrete:
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    else:
        torch.testing.assert_close(before.mean, after.mean, atol=0, rtol=0)
        torch.testing.assert_close(before.log_std, after.log_std, atol=0, rtol=0)


@pytest.mark.parametrize("discrete", [False, True])
def test_latent_only_decoder_ignores_earlier_past_but_responds_to_latent(monkeypatch, discrete):
    torch.manual_seed(17)
    model = IntentionModel(2, 3, hidden_dim=8, latent_dim=2, discrete=discrete,
                           decoder_history=False)
    if not discrete:
        # Continuous initialization intentionally predicts zero deltas. Use a
        # nonzero learned-style head to test the information path beyond init.
        with torch.no_grad():
            model.decoder_head.weight.normal_(std=0.2)
    past = torch.randn(2, 4, 2, 3)
    if discrete:
        past = torch.nn.functional.one_hot(torch.randint(3, (2, 4, 2)), 3).float()
    visible = torch.ones(2, 4, 2, dtype=torch.bool)
    if not discrete:
        visible[:, -1] = False  # Anchor is the latest visible, not final slot.
    changed_past = past.clone()
    changed_past[:, :2] = past[:, :2].roll(1, -1) if discrete else past[:, :2] + 100
    latent = torch.randn(2, 2, 2)

    def forbidden_encoder(*args, **kwargs):
        raise AssertionError("Latent-only decoding must not call the history encoder")

    monkeypatch.setattr(model, "_encode", forbidden_encoder)
    before = model.decode(past, visible, latent, horizon=3)
    after = model.decode(changed_past, visible, latent, horizon=3)
    before = before if discrete else before.mean
    after = after if discrete else after.mean
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    changed_latent = latent.clone()
    changed_latent[:, 0, 0] += 1
    changed = model.decode(past, visible, changed_latent, horizon=3)
    changed = changed if discrete else changed.mean
    assert not torch.allclose(before[:, :, 0], changed[:, :, 0])
    torch.testing.assert_close(before[:, :, 1], changed[:, :, 1], atol=0, rtol=0)


@pytest.mark.parametrize("discrete", [False, True])
def test_latent_only_reconstruction_trains_encoder_through_latent(discrete):
    torch.manual_seed(64)
    model = IntentionModel(2, 3, hidden_dim=8, latent_dim=2, discrete=discrete,
                           decoder_history=False)
    if not discrete:
        with torch.no_grad():
            model.decoder_head.weight.normal_(std=0.2)
    actions = torch.randn(2, 6, 2, 3)
    if discrete:
        actions = torch.nn.functional.one_hot(torch.randint(3, (2, 6, 2)), 3).float()
    visible = torch.ones(2, 6, 2, dtype=torch.bool)
    # Disable the KL contribution so a nonzero encoder gradient must come
    # through posterior sampling and reconstruction, not regularization.
    terms = model.loss(actions[:, :3], visible[:, :3], actions[:, 3:], visible[:, 3:],
                       beta=0.0, samples=2)
    assert all(torch.isfinite(value) for value in terms.values())
    terms["loss"].backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    for parameter in (model.encoder.weight_ih_l0, model.gaussian_head.weight,
                      model.decoder_start.weight, model.decoder_cell.weight_ih):
        assert parameter.grad.norm() > 0
