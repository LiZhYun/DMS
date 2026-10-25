import math

import pytest
import torch

from dms.diagnostics import forecast_diagnostics, summarize_forecasts
from dms.intention import DiagonalGaussian, IntentionModel


def test_continuous_metrics_use_consecutive_visible_endpoints_and_sensed_components():
    model = IntentionModel(2, 3, hidden_dim=8, latent_dim=2, action_dims=[2, 3],
                           observation_mask=[[True, False, True], [True, True, False]])
    past = torch.tensor([[[[1., 900., 900.], [3., 4., 900.]],
                          [[2., 900., 900.], [900., 900., 900.]],
                          [[900., 900., 900.], [5., 6., 900.]]]])
    visible = torch.tensor([[[True, True], [True, False], [False, True]]])
    future = torch.tensor([[[[900., 900., 900.], [8., 10., 900.]],
                            [[4., 900., 900.], [900., 900., 900.]],
                            [[900., 900., 900.], [900., 900., 900.]]]])
    future_visible = torch.tensor([[[True, True], [True, False], [False, True]]])
    result = forecast_diagnostics(model, past, visible, future, future_visible)
    assert result["forecast_target_count"] == 2
    assert result["forecast_comparison_count"] == 2
    assert result["forecast_component_count"] == 3
    # Initial decoder predicts zero deltas, hence precisely matches persistence.
    assert result["forecast_rmse"] == pytest.approx(math.sqrt(29 / 3))
    assert result["persistence_rmse"] == pytest.approx(math.sqrt(29 / 3))
    assert result["history_mean_rmse"] == pytest.approx(math.sqrt(47.25 / 3))
    assert result["history_baseline_target_count"] == 2
    assert result["history_baseline_component_count"] == 3
    assert result["history_baseline_coverage_fraction"] == 1.0
    past_mask = visible.unsqueeze(-1) & model.action_mask
    future_mask = future_visible.unsqueeze(-1) & model.action_mask
    changed = forecast_diagnostics(model, past.masked_fill(~past_mask, float("nan")), visible,
                                   future.masked_fill(~future_mask, float("nan")), future_visible)
    assert changed == result


def test_discrete_nll_uses_all_targets_but_accuracy_compares_only_known_past_actions():
    model = IntentionModel(2, 3, hidden_dim=8, latent_dim=2, discrete=True, action_dims=[2, 3])
    with torch.no_grad():
        model.decoder_head.weight.zero_()
        model.decoder_head.bias.copy_(torch.tensor([0.6, 0.3, 0.1]).log())
    past_labels = torch.tensor([[[1, 2], [0, 0]]])
    past = torch.nn.functional.one_hot(past_labels, 3).float()
    visible = torch.tensor([[[True, False], [False, False]]])
    future_labels = torch.tensor([[[1, 2], [0, 1]]])
    future = torch.nn.functional.one_hot(future_labels, 3).float()
    future_visible = torch.ones(1, 2, 2, dtype=torch.bool)
    result = forecast_diagnostics(model, past, visible, future, future_visible)
    assert result["forecast_target_count"] == 4
    assert result["forecast_comparison_count"] == 2
    expected_nll = -sum(math.log(p) for p in [1 / 3, 0.1, 2 / 3, 0.3]) / 4
    assert result["forecast_nll"] == pytest.approx(expected_nll)
    assert result["forecast_accuracy"] == 0.5
    assert result["persistence_accuracy"] == 0.5
    assert result["history_histogram_accuracy"] == 0.5
    assert result["history_histogram_nll"] == pytest.approx(-math.log(2 / 9) / 2)
    assert result["forecast_history_comparison_nll"] == pytest.approx(-math.log(2 / 9) / 2)
    assert result["history_baseline_target_count"] == 2
    assert result["history_baseline_coverage_fraction"] == 0.5
    assert "persistence_nll" not in result


@pytest.mark.parametrize("discrete", [False, True])
def test_future_targets_cannot_enter_prior_or_decoder(monkeypatch, discrete):
    torch.manual_seed(5)
    model = IntentionModel(1, 2, hidden_dim=8, latent_dim=2, discrete=discrete)
    past = torch.tensor([[[[1., 0.]], [[0., 1.]]]])
    visible = torch.ones(1, 2, 1, dtype=torch.bool)
    future = torch.tensor([[[[1., 0.]], [[1., 0.]]]])
    future_visible = torch.ones(1, 2, 1, dtype=torch.bool)
    original_decode = model.decode
    predictions = []

    def checked_decode(decoder_past, decoder_visible, latent, horizon):
        assert not torch.is_grad_enabled()
        torch.testing.assert_close(decoder_past, past)
        torch.testing.assert_close(decoder_visible, visible)
        torch.testing.assert_close(latent, model.prior(past, visible).mean)
        prediction = original_decode(decoder_past, decoder_visible, latent, horizon)
        predictions.append(prediction.mean.clone() if isinstance(prediction, DiagonalGaussian)
                           else prediction.clone())
        return prediction

    def forbidden_posterior(*args, **kwargs):
        raise AssertionError("Diagnostics must never use the posterior")

    monkeypatch.setattr(model, "decode", checked_decode)
    monkeypatch.setattr(model, "posterior", forbidden_posterior)
    rng = torch.random.get_rng_state().clone()
    before = forecast_diagnostics(model, past, visible, future, future_visible)
    after = forecast_diagnostics(model, past, visible, future.flip(-1), future_visible)
    torch.testing.assert_close(predictions[0], predictions[1], atol=0, rtol=0)
    torch.testing.assert_close(rng, torch.random.get_rng_state(), atol=0, rtol=0)
    assert before != after  # Targets change the scores, never the forecast.
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("discrete", [False, True])
def test_missing_targets_return_empty_metrics(discrete):
    model = IntentionModel(1, 2, hidden_dim=8, latent_dim=2, discrete=discrete)
    past = torch.zeros(1, 2, 1, 2)
    visible = torch.zeros(1, 2, 1, dtype=torch.bool)
    future = torch.full((1, 3, 1, 2), float("nan"))
    future_visible = torch.zeros(1, 3, 1, dtype=torch.bool)
    assert forecast_diagnostics(model, past, visible, future, future_visible) == {}
    assert forecast_diagnostics(model, past, visible, future[:, :0], future_visible[:, :0]) == {}


@pytest.mark.parametrize("discrete", [False, True])
def test_no_past_observations_never_invents_a_persistence_baseline(discrete):
    model = IntentionModel(1, 2, hidden_dim=8, latent_dim=2, discrete=discrete)
    past = torch.zeros(1, 0, 1, 2)
    visible = torch.zeros(1, 0, 1, dtype=torch.bool)
    future = torch.tensor([[[[1., 0.]], [[0., 1.]]]])
    future_visible = torch.ones(1, 2, 1, dtype=torch.bool)
    result = forecast_diagnostics(model, past, visible, future, future_visible)
    assert result["forecast_target_count"] == (2 if discrete else 1)
    assert result["forecast_comparison_count"] == 0
    assert result["history_baseline_target_count"] == 0
    assert result["history_baseline_coverage_fraction"] == 0.0
    assert not any(key.startswith("persistence_") for key in result)
    assert "history_mean_rmse" not in result
    assert "history_histogram_accuracy" not in result
    assert "history_histogram_nll" not in result
    if discrete:
        assert math.isfinite(result["forecast_nll"])
        assert "forecast_accuracy" not in result
    else:
        assert result["forecast_component_count"] == 0
        assert "forecast_rmse" not in result


def test_continuous_isolated_observation_is_not_a_valid_delta_target():
    model = IntentionModel(1, 1, hidden_dim=8, latent_dim=2)
    past = torch.zeros(1, 1, 1, 1)
    visible = torch.zeros(1, 1, 1, dtype=torch.bool)
    future = torch.ones(1, 1, 1, 1)
    future_visible = torch.ones(1, 1, 1, dtype=torch.bool)
    assert forecast_diagnostics(model, past, visible, future, future_visible) == {}


def test_summarize_forecasts_pools_squared_errors_by_component_count():
    rows = [
        {"forecast_target_count": 2, "forecast_comparison_count": 1, "forecast_component_count": 1,
         "forecast_rmse": 1., "persistence_rmse": 2.},
        {},
        {"forecast_target_count": 4, "forecast_comparison_count": 2, "forecast_component_count": 3,
         "forecast_rmse": 3., "persistence_rmse": 4.},
        {"forecast_target_count": 1, "forecast_comparison_count": 0, "forecast_component_count": 0},
    ]
    result = summarize_forecasts(iter(rows))
    assert result["forecast_target_count"] == 7
    assert result["forecast_comparison_count"] == 3
    assert result["forecast_component_count"] == 4
    assert result["forecast_rmse"] == pytest.approx(math.sqrt(7))
    assert result["persistence_rmse"] == pytest.approx(math.sqrt(13))


def test_summarize_forecasts_weights_nll_and_accuracy_by_their_separate_counts():
    rows = [
        {"forecast_target_count": 2, "forecast_comparison_count": 1,
         "forecast_nll": 1., "forecast_accuracy": 1., "persistence_accuracy": 0.},
        {"forecast_target_count": 6, "forecast_comparison_count": 3,
         "forecast_nll": 3., "forecast_accuracy": 1 / 3, "persistence_accuracy": 2 / 3},
        {"forecast_target_count": 2, "forecast_comparison_count": 0, "forecast_nll": 5.},
    ]
    result = summarize_forecasts(rows)
    assert result == {"forecast_target_count": 10, "forecast_comparison_count": 4,
                      "forecast_nll": 3., "forecast_accuracy": 0.5, "persistence_accuracy": 0.5}
    assert summarize_forecasts([]) == {}
    assert summarize_forecasts([{}, {}]) == {}


def test_history_mean_can_explain_noise_smoothing_without_any_learning():
    model = IntentionModel(1, 1, hidden_dim=8, latent_dim=2)
    past = torch.tensor([[[[-1.]], [[1.]]]])
    visible = torch.ones(1, 2, 1, dtype=torch.bool)
    future = torch.zeros(1, 2, 1, 1)
    result = forecast_diagnostics(model, past, visible, future, visible)
    assert result["forecast_rmse"] == 1.0
    assert result["persistence_rmse"] == 1.0
    assert result["history_mean_rmse"] == 0.0


def test_history_histogram_uses_only_visible_past_and_smooths_only_real_categories():
    model = IntentionModel(1, 4, hidden_dim=8, latent_dim=2, discrete=True, action_dims=[3])
    past = torch.nn.functional.one_hot(torch.tensor([[[0], [2], [0], [1]]]), 4).float()
    visible = torch.tensor([[[True], [False], [True], [True]]])
    past[:, 1] = float("nan")
    past[..., 3] = float("nan")  # Padded category must receive no pseudocount.
    future = torch.nn.functional.one_hot(torch.tensor([[[0], [2]]]), 4).float()
    future_visible = torch.ones(1, 2, 1, dtype=torch.bool)
    result = forecast_diagnostics(model, past, visible, future, future_visible)
    # Visible counts [2,1,0] become [3,2,1]/6, independent of future labels.
    assert result["history_histogram_accuracy"] == 0.5
    assert result["persistence_accuracy"] == 0.0
    assert result["history_histogram_nll"] == pytest.approx(-math.log(1 / 12) / 2)
    changed_future = torch.nn.functional.one_hot(torch.tensor([[[1], [1]]]), 4).float()
    changed = forecast_diagnostics(model, past, visible, changed_future, future_visible)
    assert changed["history_histogram_accuracy"] == 0.0
    assert changed["history_histogram_nll"] == pytest.approx(math.log(3))


@pytest.mark.parametrize("discrete", [False, True])
def test_history_baseline_aggregation_preserves_matched_coverage(discrete):
    rows = [
        {"forecast_target_count": 2, "history_baseline_target_count": 1,
         "history_baseline_coverage_fraction": 0.5},
        {"forecast_target_count": 6, "history_baseline_target_count": 2,
         "history_baseline_coverage_fraction": 1 / 3},
    ]
    if discrete:
        rows[0].update(history_histogram_nll=1., forecast_history_comparison_nll=2.,
                       history_histogram_accuracy=1.)
        rows[1].update(history_histogram_nll=4., forecast_history_comparison_nll=5.,
                       history_histogram_accuracy=0.)
    else:
        rows[0].update(history_baseline_component_count=2, history_mean_rmse=1.)
        rows[1].update(history_baseline_component_count=6, history_mean_rmse=3.)
    result = summarize_forecasts(rows)
    assert result["history_baseline_target_count"] == 3
    assert result["history_baseline_coverage_fraction"] == 3 / 8
    if discrete:
        assert result["history_histogram_nll"] == 3.
        assert result["forecast_history_comparison_nll"] == 4.
        assert result["history_histogram_accuracy"] == pytest.approx(1 / 3)
    else:
        assert result["history_baseline_component_count"] == 8
        assert result["history_mean_rmse"] == pytest.approx(math.sqrt(7))
