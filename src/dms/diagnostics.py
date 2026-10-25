"""Online, pre-update behavior forecasts from observed history alone.

These diagnostics assess newly collected suffixes before fitting them. They
are not held-out benchmark results and do not use the training posterior.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

import torch
from torch import Tensor

from dms.intention import DiagonalGaussian, IntentionModel


@torch.no_grad()
def forecast_diagnostics(
    model: IntentionModel,
    past: Tensor,
    visible: Tensor,
    future: Tensor,
    future_visible: Tensor,
) -> dict[str, float | int]:
    """Score a deterministic prior forecast before any auxiliary update.

    Inputs follow IntentionModel's [batch,time,peer,action-feature] convention.
    ``forecast_target_count`` counts training-valid time/peer actions. Continuous
    targets require consecutive observed endpoints, exactly as in ``loss``.

    Persistence repeats the latest observed past action for the whole suffix;
    it never updates itself from the future targets. Comparisons require at
    least one observed past action and use identical targets for both methods.
    ``forecast_comparison_count`` counts these time/peer actions. Continuous
    RMSE averages their sensed, unpadded components, whose count is reported as
    ``forecast_component_count``. Discrete accuracies use comparison targets;
    discrete NLL uses *all* training-valid targets, including unseen peers.

    No baseline NLL is invented for deterministic categorical persistence.
    A second baseline uses only the visible past: its componentwise mean for
    continuous actions, or action-count mode for discrete actions. Categorical
    NLL uses Laplace smoothing (one pseudocount per real, unpadded category).
    These baselines share the persistence comparison mask; unknown histories
    are excluded even though smoothing could otherwise supply a uniform guess.
    ``forecast_history_comparison_nll`` puts model NLL on that same subset;
    the existing ``forecast_nll`` continues to include every valid target.
    History-baseline target/component counts and coverage are explicit.
    Metrics with no comparison samples are omitted; no valid targets gives {}.
    Counts permit weighting across observers/batches (pool squared RMSE before
    taking its square root). The caller must invoke this before fitting suffixes.
    """
    model._check(past, visible)
    model._check(future, future_visible)
    if past.shape[0] != future.shape[0]:
        raise ValueError("past and future must have the same batch size")
    if future.shape[1] == 0:
        return {}

    target_visible = model._reconstruction_visibility(visible, future_visible)
    target_visible = target_visible & model.action_mask.any(dim=-1)
    target_count = int(target_visible.sum().item())
    if target_count == 0:
        return {}

    prior = model.prior(past, visible)
    prediction = model.decode(past, visible, prior.mean, horizon=future.shape[1])
    known_past = visible.bool().any(dim=1) & model.action_mask.any(dim=-1)
    comparison = target_visible & known_past[:, None]
    comparison_count = int(comparison.sum().item())
    result: dict[str, float | int] = {
        "forecast_target_count": target_count,
        "forecast_comparison_count": comparison_count,
        "history_baseline_target_count": comparison_count,
        "history_baseline_coverage_fraction": comparison_count / target_count,
    }
    target = model._masked_actions(future, target_visible)
    persistence = model._last_visible_action(past, visible)[:, None].expand_as(future)
    clean_past = model._masked_actions(past, visible)

    if model.discrete:
        if not isinstance(prediction, Tensor):
            raise TypeError("Discrete prediction must be logits")
        result["forecast_nll"] = float(
            model.reconstruction_nll(prediction, future, target_visible).item()
        )
        if comparison_count:
            labels = target.argmax(dim=-1)
            result["forecast_accuracy"] = float(
                (prediction.argmax(dim=-1)[comparison] == labels[comparison]).float().mean().item()
            )
            result["persistence_accuracy"] = float(
                (persistence.argmax(dim=-1)[comparison] == labels[comparison]).float().mean().item()
            )
            counts = clean_past.sum(dim=1)
            mode = counts.argmax(-1)[:, None].expand_as(labels)
            result["history_histogram_accuracy"] = float(
                (mode[comparison] == labels[comparison]).float().mean().item()
            )
            # Padded categories have zero count and zero probability. Taking
            # log of smoothed counts gives logits with the correct normalizer.
            histogram_logits = (counts + model.action_mask).log()[:, None].expand_as(future)
            result["history_histogram_nll"] = float(
                model.reconstruction_nll(histogram_logits, future, comparison).item()
            )
            result["forecast_history_comparison_nll"] = float(
                model.reconstruction_nll(prediction, future, comparison).item()
            )
    else:
        if not isinstance(prediction, DiagonalGaussian):
            raise TypeError("Continuous prediction must be a DiagonalGaussian")
        components = comparison.unsqueeze(-1) & model.action_mask
        component_count = int(components.sum().item())
        result["forecast_component_count"] = component_count
        result["history_baseline_component_count"] = component_count
        if component_count:
            result["forecast_rmse"] = float(
                (prediction.mean[components] - target[components]).square().mean().sqrt().item()
            )
            result["persistence_rmse"] = float(
                (persistence[components] - target[components]).square().mean().sqrt().item()
            )
            sensed = visible.bool().unsqueeze(-1) & model.action_mask
            history_mean = clean_past.sum(dim=1) / sensed.sum(dim=1).clamp_min(1)
            history_mean = history_mean[:, None].expand_as(future)
            result["history_mean_rmse"] = float(
                (history_mean[components] - target[components]).square().mean().sqrt().item()
            )
    return result


def summarize_forecasts(
    rows: Iterable[Mapping[str, float | int]],
) -> dict[str, float | int]:
    """Pool diagnostic rows using each metric's actual sample count.

    Counts are summed. NLL is weighted by valid target actions, accuracy by
    comparable actions, and RMSE by comparable action components. Squared RMSE
    is pooled before taking the square root; averaging RMSE values is incorrect.
    Empty rows and absent comparison metrics contribute no samples.
    """
    rows = list(rows)
    result: dict[str, float | int] = {}
    for count in ("forecast_target_count", "forecast_comparison_count", "forecast_component_count",
                  "history_baseline_target_count", "history_baseline_component_count"):
        if any(count in row for row in rows):
            result[count] = sum(int(row.get(count, 0)) for row in rows)

    weights = {
        "forecast_nll": "forecast_target_count",
        "forecast_accuracy": "forecast_comparison_count",
        "persistence_accuracy": "forecast_comparison_count",
        "forecast_rmse": "forecast_component_count",
        "persistence_rmse": "forecast_component_count",
        "history_mean_rmse": "history_baseline_component_count",
        "history_histogram_accuracy": "history_baseline_target_count",
        "history_histogram_nll": "history_baseline_target_count",
        "forecast_history_comparison_nll": "history_baseline_target_count",
    }
    for metric, count in weights.items():
        samples = [(float(row[metric]), int(row[count])) for row in rows if metric in row]
        total = sum(weight for _, weight in samples)
        if total:
            is_rmse = metric.endswith("_rmse")
            average = sum((value * value if is_rmse else value) * weight
                          for value, weight in samples) / total
            result[metric] = math.sqrt(average) if is_rmse else average
    if "history_baseline_target_count" in result and result.get("forecast_target_count", 0):
        result["history_baseline_coverage_fraction"] = (
            result["history_baseline_target_count"] / result["forecast_target_count"]
        )
    return result
