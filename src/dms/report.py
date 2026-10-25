"""Aggregate final evaluations over independent training seeds.

Episode returns are averaged inside each seed before estimating uncertainty.
A published point-reference gate is deliberately separate from a claim that
the paper's experiments, environment, and mechanism have been reproduced.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _above(value: float, threshold: float) -> bool:
    # Summing identical decimal win rates can round a nominal tie upward.
    return bool(value > threshold and not np.isclose(value, threshold, rtol=0, atol=1e-12))


def _valid_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdefABCDEF" for character in value))


def _environment_key(environment: Any) -> Any:
    """Ignore only a resolved, content-identified map's installation location.

    Keep the original record intact. An unresolved asset, missing/invalid hash,
    relative map name, build selection, and every other setting remain strict.
    Runtime counters belong in the separate runtime record, never here.
    """
    result = deepcopy(environment)
    if isinstance(result, dict):
        asset = result.get("map_asset")
        if (isinstance(asset, dict) and asset.get("status") == "resolved"
                and _valid_sha256(asset.get("sha256"))):
            asset.pop("path", None)
    return result


def _required_packages(config: Any, environment: Any) -> tuple[str, ...]:
    packages = ("torch", "numpy")
    config = config if isinstance(config, dict) else {}
    environment = environment if isinstance(environment, dict) else {}
    if config.get("env") == "smac":
        return packages + ("pysc2", "smac", "s2clientprotocol", "protobuf")
    if config.get("env") == "mamujoco":
        legacy = (config.get("mamujoco_backend") == "legacy"
                  or "historical" in str(environment.get("benchmark_track", "")).lower())
        return packages + (("gym", "mujoco-py") if legacy else
                           ("gymnasium", "gymnasium-robotics", "mujoco"))
    return packages


def _version_key(config: Any, versions: Any, environment: Any) -> dict[str, Any]:
    """Compare the packages actually used, with old backend metadata fallback.

    Old training inventories omit some packages that their environment already
    records. Conflicting declarations are retained as a conflict, not silently
    resolved in favor of either source. Full inventories stay in raw records.
    """
    inventory = versions if isinstance(versions, dict) else {}
    backend = environment.get("packages", {}) if isinstance(environment, dict) else {}
    backend = backend if isinstance(backend, dict) else {}
    selected = {}
    for package in _required_packages(config, environment):
        if package in inventory and package in backend and inventory[package] != backend[package]:
            selected[package] = {"inventory": inventory[package], "environment": backend[package]}
        else:
            selected[package] = inventory.get(package, backend.get(package))
    return selected


def _validate_evaluation(record: dict[str, Any]) -> dict[str, Any]:
    seed = _integer(record.get("training_seed"), "training_seed")
    _integer(record.get("training_steps"), "training_steps", 1)
    episodes = _integer(record.get("episodes"), "episodes", 1)
    rows = record.get("results")
    if not isinstance(rows, list) or len(rows) != episodes:
        raise ValueError(f"Seed {seed}: episodes must equal the number of raw results")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Seed {seed}: raw episode results must be objects")
    episode_seeds = [_integer(row.get("seed"), "episode seed") for row in rows]
    if len(set(episode_seeds)) != episodes:
        raise ValueError(f"Seed {seed}: duplicate evaluation episode seeds")
    if "episode_seed_start" in record:
        start = _integer(record["episode_seed_start"], "episode_seed_start")
        if episode_seeds != list(range(start, start + episodes)):
            raise ValueError(f"Seed {seed}: raw episode seeds do not match episode_seed_start")
    returns = [_finite(row.get("return"), "episode return") for row in rows]
    mean_return = float(np.mean(returns))
    reported_return = _finite(record.get("return_mean"), "return_mean")
    if not np.isclose(mean_return, reported_return, rtol=1e-8, atol=1e-8):
        raise ValueError(f"Seed {seed}: return_mean disagrees with raw episode returns")
    result = {"training_seed": seed, "return_mean": mean_return}
    if "win_rate" in record:
        wins = [row.get("battle_won") for row in rows]
        if not all(isinstance(win, bool) for win in wins):
            raise ValueError(f"Seed {seed}: win_rate requires boolean battle_won outcomes")
        win_rate = _finite(record["win_rate"], "win_rate")
        if not 0 <= win_rate <= 1 or not np.isclose(
            win_rate, np.mean(wins), rtol=1e-8, atol=1e-8
        ):
            raise ValueError(f"Seed {seed}: win_rate disagrees with raw battle outcomes")
        result["win_rate"] = win_rate
    config = record.get("config")
    if config is not None:
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        if "seed" in config and config["seed"] != seed:
            raise ValueError(f"Seed {seed}: config.seed disagrees with training_seed")
        if "total_steps" in config:
            planned = _integer(config["total_steps"], "config.total_steps", 1)
            if record["training_steps"] > planned:
                raise ValueError(f"Seed {seed}: completed steps exceed the declared training budget")
    return result


def _consistency_key(record: dict[str, Any]) -> dict[str, Any]:
    # Paths, training RNG seeds, evaluation seeds and device placement can
    # change between independent replicates; the experiment itself cannot.
    config = record.get("config")
    if isinstance(config, dict):
        config = {
            key: value for key, value in config.items()
            if key not in {"seed", "output", "device", "eval_seed"}
        }
    manifest = record.get("manifest")
    if manifest is not None and not isinstance(manifest, dict):
        raise ValueError("manifest must be an object or null")
    return {
        "config": config,
        "training_steps": record["training_steps"],
        "evaluation_episodes": record["episodes"],
        "has_win_rate": "win_rate" in record,
        "environment_metadata": _environment_key(manifest.get("environment")) if manifest else None,
        "dependency_versions": _version_key(record.get("config"), manifest.get("versions"),
                                             manifest.get("environment")) if manifest else None,
        "evaluation_protocol": record.get("evaluation_protocol"),
    }


def _source_provenance(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Allow legacy summaries, but never certify missing or changed source.

    Different declared training snapshots are different experiments. Runtime
    hosts, Slurm jobs and devices are intentionally irrelevant to this check.
    """
    training = [(record.get("manifest") or {}).get("source_sha256") for record in records]
    declared = [source for source in training if source not in (None, {})]
    if declared and any(source != declared[0] for source in declared[1:]):
        raise ValueError("Mismatched training source snapshots: manifest.source_sha256 differs across seeds")

    def valid(source):
        return (
            isinstance(source, dict) and bool(source)
            and all(isinstance(path, str) and path.startswith("dms/") and path.endswith(".py")
                    and isinstance(digest, str) and len(digest) == 64
                    and all(character in "0123456789abcdefABCDEF" for character in digest)
                    for path, digest in source.items())
        )

    warnings, matches = [], []
    for record, trained_source in zip(records, training):
        evaluated_source = record.get("evaluation_source_sha256")
        training_valid, evaluation_valid = valid(trained_source), valid(evaluated_source)
        for label, source, is_valid in [
            ("training", trained_source, training_valid),
            ("evaluation", evaluated_source, evaluation_valid),
        ]:
            if not is_valid:
                status = "missing" if source in (None, {}) else "invalid"
                warnings.append(f"Seed {record['training_seed']}: {status} {label} source SHA-256 mapping")
        match = training_valid and evaluation_valid and trained_source == evaluated_source
        matches.append(match)
        if training_valid and evaluation_valid and not match:
            warnings.append(f"Seed {record['training_seed']}: evaluation source differs from training snapshot")
    return {
        "training_sources_complete": all(valid(source) for source in training),
        "evaluation_sources_match_training": all(matches),
        "training_source_sha256": declared[0] if declared and valid(declared[0]) else None,
        "warnings": warnings,
        "interpretation": "Hashes identify recorded source files; they do not independently prove execution or benchmark equivalence.",
    }


def _environment_provenance(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Missing or changed evaluation evidence blocks certification, not summaries."""
    warnings = []
    training_complete, environments_match, versions_complete, versions_match = [], [], [], []
    for record in records:
        seed, config = record["training_seed"], record.get("config")
        manifest = record.get("manifest") or {}
        trained = manifest.get("environment")
        evaluated = record.get("evaluation_environment")

        def valid_environment(value):
            return (isinstance(value, dict) and bool(value)
                    and isinstance(value.get("benchmark_track"), str) and bool(value["benchmark_track"]))

        trained_valid, evaluated_valid = valid_environment(trained), valid_environment(evaluated)
        training_complete.append(trained_valid)
        match = trained_valid and evaluated_valid and _environment_key(trained) == _environment_key(evaluated)
        environments_match.append(match)
        for label, valid in [("training", trained_valid), ("evaluation", evaluated_valid)]:
            if not valid:
                warnings.append(f"Seed {seed}: missing or invalid {label} environment metadata")
        if trained_valid and evaluated_valid and not match:
            warnings.append(f"Seed {seed}: evaluation environment differs from training environment")

        selected = []
        valid_versions = []
        for label, inventory, environment in [
            ("training", manifest.get("versions"), trained),
            ("evaluation", record.get("evaluation_versions"), evaluated),
        ]:
            values = _version_key(config, inventory, environment)
            selected.append(values)
            valid = isinstance(inventory, dict) and bool(inventory)
            missing = [name for name, version in values.items() if not isinstance(version, str) or not version]
            valid = valid and not missing
            valid_versions.append(valid)
            if not valid:
                detail = ", ".join(missing) if missing else "version inventory"
                warnings.append(f"Seed {seed}: missing, invalid, or conflicting {label} dependency versions: {detail}")
        versions_complete.append(valid_versions[0])
        version_match = all(valid_versions) and selected[0] == selected[1]
        versions_match.append(version_match)
        if all(valid_versions) and not version_match:
            changed = sorted(name for name in set(selected[0]) | set(selected[1])
                             if selected[0].get(name) != selected[1].get(name))
            warnings.append(f"Seed {seed}: evaluation dependency versions differ from training: {', '.join(changed)}")
    return {
        "training_environments_complete": all(training_complete),
        "evaluation_environments_match_training": all(environments_match),
        "training_dependency_versions_complete": all(versions_complete),
        "evaluation_dependency_versions_match_training": all(versions_match),
        "warnings": warnings,
    }


def seed_statistics(
    values: list[float], *, bootstrap_samples: int = 10000, bootstrap_seed: int = 0
) -> dict[str, Any]:
    """Percentile bootstrap of the mean, sampling whole training-seed means."""
    _integer(bootstrap_samples, "bootstrap_samples", 1)
    _integer(bootstrap_seed, "bootstrap_seed")
    if not values:
        raise ValueError("At least one independent training seed is required")
    array = np.array([_finite(v, "seed value") for v in values], dtype=np.float64)
    interval = None
    if len(array) > 1:
        rng = np.random.default_rng(bootstrap_seed)
        draws = rng.choice(array, size=(bootstrap_samples, len(array)), replace=True)
        interval = np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else None,
        "bootstrap_mean_95_percent_interval": interval,
        "independent_training_seeds": len(array),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_rng_seed": bootstrap_seed,
        "resampling_unit": "independent_training_seed_mean",
    }


def _target(target_id: str, targets: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for row in targets.get("smac", []):
        if row["id"] == target_id:
            return "smac", row
    for row in targets.get("mamujoco", []):
        if row["id"] == target_id:
            return "mamujoco", {**targets["mamujoco_shared"], **row}
    raise ValueError(f"Unknown primary paper target {target_id!r}")


def _comparison_metadata(record: dict[str, Any], family: str, target: dict[str, Any]):
    config, manifest = record.get("config"), record.get("manifest")
    if not isinstance(config, dict) or not config:
        raise ValueError("Paper comparison requires nonempty config metadata")
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Paper comparison requires nonempty manifest metadata")
    environment, versions = manifest.get("environment"), manifest.get("versions")
    if not isinstance(environment, dict) or not environment.get("benchmark_track"):
        raise ValueError("Paper comparison requires manifest.environment.benchmark_track")
    if not isinstance(versions, dict) or not versions:
        raise ValueError("Paper comparison requires dependency version metadata")
    if config.get("env") != family or not config.get("algorithm"):
        raise ValueError("Paper comparison task/environment/algorithm metadata is incompatible")
    if config.get("seed") != record["training_seed"]:
        raise ValueError("Paper comparison requires config.seed matching training_seed")
    _integer(config.get("total_steps"), "config.total_steps", 1)
    protocol = record.get("evaluation_protocol")
    if not isinstance(protocol, dict) or protocol.get("deterministic") is not True:
        raise ValueError("Paper comparison requires declared deterministic evaluation metadata")
    if family == "smac":
        if config.get("map_name") != target["id"] or environment.get("map_name") != target["id"]:
            raise ValueError("Paper target does not match the evaluated SMAC map")
        required = ["game_version", "difficulty"]
    else:
        expected = {"scenario": target["environment"], "agent_conf": target["agent_configuration"]}
        if any(config.get(k) != v or environment.get(k) != v for k, v in expected.items()):
            raise ValueError("Paper target does not match the evaluated MuJoCo task/partition")
        required = ["agent_obsk"]
        if config.get("agent_obsk") != environment.get("agent_obsk"):
            raise ValueError("MuJoCo config and environment observation radius disagree")
    if any(key not in environment or environment[key] is None for key in required):
        raise ValueError("Paper comparison lacks required environment settings")
    if not (environment.get("action_sensor") or environment.get("action_observability")):
        raise ValueError("Paper comparison requires the local action-sensor declaration")
    return config, environment


def _computed_ant_forces(record: dict[str, Any]) -> bool:
    """Recognize an override even in an old or incomplete provenance record."""
    if (record.get("config") or {}).get("ant_contact_forces") == "computed":
        return True
    for metadata in ((record.get("manifest") or {}).get("environment"),
                     record.get("evaluation_environment")):
        if not isinstance(metadata, dict):
            continue
        force = metadata.get("contact_force_computation")
        if (metadata.get("ant_contact_forces") == "computed"
                or isinstance(force, dict) and force.get("mode") == "computed"
                or "computed_forces" in str(metadata.get("benchmark_track", ""))):
            return True
    return False


def _compare(
    records: list[dict[str, Any]], report: dict[str, Any],
    target_id: str, targets: dict[str, Any],
) -> dict[str, Any]:
    family, target = _target(target_id, targets)
    config, environment = _comparison_metadata(records[0], family, target)
    for record in records[1:]:
        _comparison_metadata(record, family, target)
    metric = "win_rate" if family == "smac" else "return_mean"
    if metric not in report["metrics"]:
        raise ValueError("SMAC paper comparison requires measured win rates")
    threshold = _finite(target["acceptance"]["threshold"], "reference threshold")
    statistics = report["metrics"][metric]
    interval = statistics["bootstrap_mean_95_percent_interval"]
    policy = targets["reproduction_evaluation_policy"]
    expected_seeds = sorted(policy["training_seeds"])
    expected_episodes = policy[f"{family}_final_evaluation_episodes_per_seed"]
    checks = {
        "registered_final_training_seeds": report["training_seeds"] == expected_seeds,
        "matched_environment_steps": report["training_steps"] == target["environment_steps"],
        "declared_final_checkpoint_budget": config["total_steps"] == target["environment_steps"],
        "registered_evaluation_episode_count": report["episodes_per_training_seed"] == expected_episodes,
        "dms_algorithm": config["algorithm"] == "dms",
        "mean_exceeds_reference_point": _above(statistics["mean"], threshold),
        "training_source_provenance_complete": report["provenance"]["training_sources_complete"],
        "evaluation_source_matches_training": report["provenance"]["evaluation_sources_match_training"],
        "training_environment_provenance_complete": report["provenance"]["training_environments_complete"],
        "evaluation_environment_matches_training": report["provenance"]["evaluation_environments_match_training"],
        "training_dependency_versions_complete": report["provenance"]["training_dependency_versions_complete"],
        "evaluation_dependency_versions_match_training": report["provenance"]["evaluation_dependency_versions_match_training"],
    }
    # Diagnostic evaluation episodes cannot be reused as the final test set.
    disjoint = True
    for record in records:
        cfg = record["config"]
        if "eval_seed" not in cfg or "eval_episodes" not in cfg:
            disjoint = False
            continue
        diagnostic = set(range(int(cfg["eval_seed"]), int(cfg["eval_seed"]) + int(cfg["eval_episodes"])))
        disjoint = disjoint and not diagnostic.intersection(row["seed"] for row in record["results"])
    checks["final_episode_seeds_disjoint_from_diagnostics"] = disjoint
    # Explicit force recomputation changes Ant observations and its contact
    # penalty. Keep the numerical reference comparison, but do not certify an
    # unverified simulator modification as a registered benchmark result.
    computed_ant = (family == "mamujoco" and target["environment"] == "Ant"
                    and any(_computed_ant_forces(record) for record in records))
    checks["no_unverified_contact_force_override"] = not computed_ant
    modern = family == "mamujoco" and "modern" in environment["benchmark_track"].lower()
    if family == "smac":
        scope = "published_point_under_declared_evaluation"
        interpretation = (
            "Exceedance concerns the published point under the declared evaluation. "
            "The paper's median aggregation and raw sampling uncertainty are unresolved; "
            "this is not an exact statistical reproduction of its raw results."
        )
    elif computed_ant:
        scope = "computed_ant_contact_forces_adaptation_reference_only"
        interpretation = (
            "Explicit Ant force recomputation changes local force observations and the contact penalty. "
            "Historical environment equivalence has not been established. The manuscript plot threshold "
            "is retained only as an engineering reference; this variant cannot pass the release reference gate."
        )
    else:
        scope = "modern_adaptation_reference_only" if modern else "unverified_mujoco_reference_only"
        interpretation = (
            "The threshold is an engineering target from approximate manuscript plot readings. "
            "Modern MaMuJoCo v5 and original v2 are separate benchmark tracks. "
            "Beating this reference does not establish improvement in the original environment."
        )
    return {
        "target_id": target_id,
        "metric": metric,
        "threshold": threshold,
        "operator": ">",
        "source_kind": target["source_kind"],
        "source_label": target.get("source_label"),
        "source_path": target.get("source_path"),
        "threshold_kind": "exact_published_point" if family == "smac" else target["acceptance_threshold_kind"],
        "scope": scope,
        "interpretation": interpretation,
        "mean_difference_from_reference": statistics["mean"] - threshold,
        "mean_exceeds_reference_point": checks["mean_exceeds_reference_point"],
        "median_exceeds_reference_point": _above(statistics["median"], threshold),
        "bootstrap_interval_above_reference": bool(interval is not None and _above(interval[0], threshold)),
        "uncertainty_limit": "Interval describes new independent training-seed means, not uncertainty of the old paper estimate.",
        "release_reference_gate": all(checks.values()),
        "release_reference_checks": checks,
        "failed_release_reference_checks": [key for key, value in checks.items() if not value],
        "paper_reproduction_pass": False,
        "paper_reproduction_reason": "An aggregate alone cannot verify benchmark equivalence, intended mechanism, ablations, or the paper's unresolved evaluation protocol.",
    }


def aggregate_evaluations(
    evaluations: list[dict[str, Any]], *, target_id: str | None = None,
    targets: dict[str, Any] | None = None,
    bootstrap_samples: int = 10000, bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Validate final evaluation JSON objects, then aggregate equal-weight seeds."""
    if not evaluations or not all(isinstance(row, dict) for row in evaluations):
        raise ValueError("Provide at least one evaluation JSON object")
    validated = [_validate_evaluation(record) for record in evaluations]
    seeds = [row["training_seed"] for row in validated]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate training seeds are not independent replicates")
    pairs = sorted(zip(evaluations, validated), key=lambda pair: pair[1]["training_seed"])
    records, per_seed = map(list, zip(*pairs))
    first_key = _consistency_key(records[0])
    for record in records[1:]:
        key = _consistency_key(record)
        differences = [name for name in first_key if key[name] != first_key[name]]
        if differences:
            raise ValueError(f"Mismatched experiment metadata: {', '.join(differences)}")
    metrics = {
        key: seed_statistics([row[key] for row in per_seed],
                             bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
        for key in per_seed[0] if key != "training_seed"
    }
    source_provenance = _source_provenance(records)
    environment_provenance = _environment_provenance(records)
    warnings = source_provenance["warnings"] + environment_provenance["warnings"]
    report = {
        "schema_version": 1,
        "training_seeds": sorted(seeds),
        "training_steps": records[0]["training_steps"],
        "episodes_per_training_seed": records[0]["episodes"],
        "experiment": first_key,
        "per_seed": per_seed,
        "metrics": metrics,
        "provenance": {**source_provenance, **environment_provenance, "warnings": warnings},
        "recorded_provenance": [deepcopy({
            "training_seed": record["training_seed"], "manifest": record.get("manifest"),
            "evaluation_environment": record.get("evaluation_environment"),
            "evaluation_versions": record.get("evaluation_versions"),
            "evaluation_runtime": record.get("evaluation_runtime"),
        }) for record in records],
        "paper_reproduction_pass": False,
        "comparison": None,
    }
    if target_id is not None:
        if targets is None:
            raise ValueError("Paper comparison requires the target ledger")
        report["comparison"] = _compare(records, report, target_id, targets)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--target", help="Primary task ID in paper_targets.json")
    parser.add_argument("--targets", type=Path, default=Path("configs/paper_targets.json"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    try:
        records = [json.loads(path.read_text()) for path in args.inputs]
        targets = json.loads(args.targets.read_text()) if args.target else None
        result = aggregate_evaluations(
            records, target_id=args.target, targets=targets,
            bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    result["input_files"] = [str(path.resolve()) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another report writer racing us.
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "training_seeds": result["training_seeds"]}))


if __name__ == "__main__":
    main()
