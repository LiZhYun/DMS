"""Measure history-filter KL on fresh frozen-policy trajectories; never train.

This supplemental mechanism probe reuses diagnose_intentions.collect_fresh.
It evaluates mask decisions on the same recorded histories, not returns under
retuned thresholds. Native collection requires a Slurm allocation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch

import dms
from dms.intention import gaussian_kl
from dms.train import Config, Learner, create_env, seed_everything


def _load_helpers():
    path = Path(__file__).resolve().with_name("diagnose_intentions.py")
    spec = importlib.util.spec_from_file_location("dms_history_filter_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load diagnostic collection helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_helpers()
PAPER_SETTINGS = ((100.0, 6), (10_000.0, 4), (1_000_000.0, 2))
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def summarize(values, threshold):
    """Pool eligible prefix/observer/peer entries, not independent episodes."""
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("eligible history-filter KL values must be finite")
    count = len(array)
    quantiles = dict(zip(map(str, QUANTILES), np.quantile(array, QUANTILES).tolist())) if count else None
    return {
        "eligible_count": count,
        "rejected_count": int((array > threshold).sum()),
        "rejection_fraction": float((array > threshold).mean()) if count else None,
        "kl_mean": float(array.mean()) if count else None,
        "negative_kl_count": int((array < 0).sum()),
        "kl_quantiles": quantiles,
        "development_quantile_thresholds": [
            {"quantile": float(probability), "threshold": value,
             "fraction_strictly_above": float((array > value).mean())}
            for probability, value in (quantiles or {}).items()
        ],
    }


@torch.no_grad()
def inspect_filter(model, past, visible, observer, settings=PAPER_SETTINGS):
    """Return exact mask/score records for [batch,history,peer,action] inputs.

    Eligibility requires a sensed recent action and a real visible component.
    Self rows are retained separately: the production filter also changes the
    observer's own intention query. Unknown peers never dilute peer statistics.
    """
    model._check(past, visible)
    visible = visible.bool()
    real_peer = model.action_mask.any(-1)
    if (visible & ~real_peer[None, None]).any():
        raise ValueError("visible history contradicts the model's local component mask")
    if not 0 <= observer < visible.shape[-1]:
        raise ValueError("observer must index a modeled agent")
    long = model.prior(past, visible)
    rows = []
    for threshold, length in settings:
        if length < 1 or not np.isfinite(threshold) or threshold < 0:
            raise ValueError("filter settings require positive length and finite nonnegative threshold")
        short = model.prior(past[:, -length:], visible[:, -length:])
        # Exact production direction/reduction: sum latent dimensions, never
        # divide by history length or by the number of observed actions.
        kl = gaussian_kl(long, short)
        recent = visible[:, -length:]
        eligible = recent.any(1) & real_peer
        if not torch.isfinite(kl[eligible]).all():
            raise ValueError("eligible history-filter KL values must be finite")
        expected_rejected = (kl > threshold) & eligible
        filtered, actual_rejected = model.filtered_history(past, visible, length, threshold)
        expected_mask = visible.clone()
        expected_mask[:, -length:] &= ~expected_rejected[:, None]
        if not torch.equal(actual_rejected, expected_rejected) or not torch.equal(filtered, expected_mask):
            raise RuntimeError("diagnostic decisions differ from the loaded history-filter implementation")
        for peer in range(visible.shape[-1]):
            mask = eligible[:, peer]
            values = kl[:, peer][mask].double().cpu().tolist()
            recent_entries = int(recent[:, :, peer].sum().item())
            removed_entries = int((visible[:, :, peer] & ~filtered[:, :, peer]).sum().item())
            rows.append({
                "observer": observer, "peer": peer, "self": observer == peer,
                "threshold": threshold, "short_length": length,
                "effective_short_length": min(length, past.shape[1]),
                "full_history_slots": past.shape[1],
                "recent_visible_action_entries": recent_entries,
                "removed_visible_action_entries": removed_entries,
                **summarize(values, threshold),
                "raw": {
                    "sample_indices": torch.where(mask)[0].cpu().tolist(),
                    "kl": values,
                    "recent_visible_steps": recent[:, :, peer].sum(1)[mask].cpu().tolist(),
                    "full_visible_steps": visible[:, :, peer].sum(1)[mask].cpu().tolist(),
                    "rejected": actual_rejected[:, peer][mask].cpu().tolist(),
                },
            })
    return rows


@torch.no_grad()
def diagnose(learner, data):
    """Use only raw, locally sensed past histories from the shared collector."""
    steps, environments = data["ends"].shape
    settings = list(PAPER_SETTINGS)
    configured = (float(learner.config.intention_threshold), learner.config.short_history)
    if configured not in settings:
        settings.append(configured)
    rows = []
    for observer, model in enumerate(learner.intentions):
        past = learner.tensor(data["history"][:, :, :, observer].reshape(
            steps*environments, learner.config.history_length, learner.num_agents, learner.action_dim))
        visible = learner.tensor(data["visible"][:, :, :, observer].reshape(
            steps*environments, learner.config.history_length, learner.num_agents), torch.bool)
        rows.extend(inspect_filter(model, past, visible, observer, settings))
    pooled = []
    for threshold, length in settings:
        entry = {"threshold": threshold, "short_length": length,
                 "paper_joint_setting": (threshold, length) in PAPER_SETTINGS,
                 "configured_setting": (threshold, length) == configured}
        for group, is_self in (("peers", False), ("self", True)):
            selected = [row for row in rows if row["threshold"] == threshold
                        and row["short_length"] == length and row["self"] == is_self]
            values = [value for row in selected for value in row["raw"]["kl"]]
            entry[group] = {
                **summarize(values, threshold),
                "recent_visible_action_entries": sum(row["recent_visible_action_entries"] for row in selected),
                "removed_visible_action_entries": sum(row["removed_visible_action_entries"] for row in selected),
            }
        pooled.append(entry)
    return {"per_observer_peer": rows, "settings": pooled,
            "sample_index_mapping": "sample_index = collection_step * num_environments + environment_index",
            "episode_ends": data["ends"].astype(bool).tolist()}


def source_hashes():
    sources = helpers.current_source_hashes()
    sources["scripts/diagnose_history_filter.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Keep assets explicit even if an older collection helper is selected.
    package = Path(dms.__file__).resolve().parent
    for path in (package / "assets").rglob("*.xml"):
        sources[f"dms/{path.relative_to(package)}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=128, help="fresh transitions per environment")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=400001)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("History-filter collection requires a Slurm allocation")
    if min(args.steps, args.num_envs) < 1 or args.seed < 0:
        parser.error("steps and num-envs must be positive; seed must be nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.checkpoint.stat().st_mode & 0o222:
        parser.error("Use an immutable checkpoint copy with all write permission bits removed")
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=args.device, weights_only=False)
    config = replace(Config(**checkpoint["config"]), device=args.device)
    config.validate()
    if config.algorithm != "dms":
        parser.error("History-filter diagnostics require a DMS checkpoint")
    torch.set_num_threads(config.torch_threads)
    seed_everything(args.seed)
    sources = source_hashes()
    environments = []
    try:
        for index in range(args.num_envs):
            environments.append(create_env(replace(config, seed=args.seed+index)))
        learner = Learner(config, environments[0])
        learner.agents.load_state_dict(checkpoint["agents"])
        learner.intentions.load_state_dict(checkpoint["intentions"])
        learner.agents.eval()
        learner.intentions.eval()
        learner.completed_episodes = checkpoint["episodes"]
        state_hash = helpers.learner_state_hash(learner)
        data, collection = helpers.collect_fresh(learner, environments, args.steps, args.seed)
        measurements = diagnose(learner, data)
        if helpers.learner_state_hash(learner) != state_hash or learner.completed_episodes != checkpoint["episodes"]:
            raise RuntimeError("History-filter diagnostic changed model state or training phase")
        metadata = [getattr(environment, "metadata", {}) for environment in environments]
    finally:
        for environment in environments:
            environment.close()
    if sources != source_hashes():
        raise RuntimeError("Diagnostic source or packaged XML changed during collection")
    if args.checkpoint.stat().st_mode & 0o222 or hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError("Checkpoint changed or lost read-only protection during diagnostic")
    manifest_path = args.checkpoint.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    result = {
        "purpose": "frozen_history_filter_mechanism_diagnostic_not_return_tuning",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_unchanged": True, "checkpoint_read_only": True,
        "training_steps": checkpoint["steps"], "training_seed": config.seed,
        "training_episodes": checkpoint["episodes"], "config": asdict(config),
        "checkpoint_source_sha256": manifest.get("source_sha256"),
        "checkpoint_environment": manifest.get("environment"),
        "diagnostic_source_sha256": sources, "source_and_xml_unchanged": True,
        "model_state_sha256": state_hash, "model_state_unchanged": True, "parameter_updates": 0,
        "steps_per_environment": args.steps, "num_environments": args.num_envs,
        "out_of_training_transitions": args.steps*args.num_envs, "diagnostic_seed": args.seed,
        "collection": collection, "environments": metadata,
        "filter_active_in_collected_policy": config.filter_history and collection["intentions_active"],
        "runtime": {"slurm_job_id": os.environ["SLURM_JOB_ID"], "host": platform.node(),
                    "device": args.device, "torch": torch.__version__, "numpy": np.__version__,
                    "diagnostic_package_root": str(Path(dms.__file__).resolve().parent),
                    "python_executable": sys.executable},
        "protocol": {
            "direction": "KL(full-history prior || recent-history prior); manuscript direction unspecified",
            "reduction": "sum over latent coordinates per history/observer/peer; no time or action-count division",
            "eligibility": "at least one actually observed action in recent l slots; local component mask retained",
            "rejection": "strict KL > K; mask only recent l slots; older visibility unchanged",
            "collection": "stochastic frozen checkpoint policy with its original configured filter; raw unfiltered past histories retained",
            "quantiles": "descriptive development cutoffs only; no threshold selection, alternate rollouts, or return tuning",
            "sampling": "all collected prefixes; overlapping histories and correlated entries are not independent training-seed replicates",
            "padding": "full checkpoint history window includes original unknown padding after reset; never trim it for this comparison",
        },
        **measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "transitions": args.steps*args.num_envs,
                      "peer_rejection": [{"K": row["threshold"], "l": row["short_length"],
                                          "fraction": row["peers"]["rejection_fraction"]}
                                         for row in measurements["settings"]]}))


if __name__ == "__main__":
    main()
