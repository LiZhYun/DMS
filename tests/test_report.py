"""Manufactured fixtures for aggregation behavior, never experiment results."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from dms.report import aggregate_evaluations, main, seed_statistics


@pytest.fixture
def targets():
    return json.loads((Path(__file__).parents[1] / "configs/paper_targets.json").read_text())


def fixture_evaluation(
    seed, *, env="smac", episodes=10, steps=10000000, value=1.0, wins=None
):
    """Create clearly fictional, internally consistent evaluation JSON data."""
    config = {
        "env": env, "algorithm": "dms", "seed": seed,
        "total_steps": 10000000, "output": f"fictional-fixture-{seed}",
        "eval_seed": 10001, "eval_episodes": 10,
    }
    if env == "smac":
        config["map_name"] = "3s5z"
        environment = {
            "benchmark_track": "original_smac", "map_name": "3s5z",
            "game_version": "4.10", "difficulty": "7",
            "action_sensor": "native_local_obs_last_action",
            "map_asset": {"status": "resolved", "path": "/fixture-machine-a/Maps/SMAC_Maps/3s5z.SC2Map",
                          "relative_path": "SMAC_Maps/3s5z.SC2Map", "size_bytes": 100, "sha256": "a" * 64},
            "runtime_selection": {"game_version": "4.10.0", "build_version": 75689, "data_version": "fixture"},
            "process_lifetime": {"restart_at_episode_boundary_after_game_loops": 500000,
                                 "seed_schedule": "fixture_seed_schedule"},
        }
    else:
        config.update(scenario="HalfCheetah", agent_conf="2x3", agent_obsk=1)
        environment = {
            "benchmark_track": "modern_mamujoco_adaptation", "scenario": "HalfCheetah",
            "agent_conf": "2x3", "agent_obsk": 1,
            "action_sensor": "previous_executed_control_on_observed_joints",
        }
    wins = episodes if wins is None else wins
    versions = {name: "fixture-version" for name in (
        "torch", "numpy", "gymnasium", "gymnasium-robotics", "mujoco", "gym", "mujoco-py",
        "pysc2", "smac", "s2clientprotocol", "protobuf",
    )}
    results = [
        {"seed": 500001 + i, "return": float(value), "length": 100,
         "battle_won": (i < wins) if env == "smac" else None}
        for i in range(episodes)
    ]
    record = {
        "fixture_only": True,
        "training_seed": seed, "training_steps": steps,
        "episodes": episodes, "episode_seed_start": 500001,
        "return_mean": float(value), "results": results, "config": config,
        "manifest": {
            "environment": environment, "versions": versions,
            "source_sha256": {"dms/train.py": "a" * 64, "dms/evaluate.py": "b" * 64},
        },
        "evaluation_source_sha256": {"dms/train.py": "a" * 64, "dms/evaluate.py": "b" * 64},
        "evaluation_environment": deepcopy(environment),
        "evaluation_versions": deepcopy(versions),
        "evaluation_protocol": {
            "deterministic": True, "episode_seeds": "consecutive from episode_seed_start"
        },
    }
    if env == "smac":
        record["win_rate"] = wins / episodes
    return record


def test_seed_bootstrap_does_not_treat_episodes_as_independent_training_runs():
    records = [fixture_evaluation(11, episodes=1000, value=0),
               fixture_evaluation(23, episodes=1000, value=10)]
    report = aggregate_evaluations(records, bootstrap_samples=4000, bootstrap_seed=7)
    statistics = report["metrics"]["return_mean"]
    assert statistics["mean"] == 5
    assert statistics["median"] == 5
    assert statistics["bootstrap_mean_95_percent_interval"] == [0, 10]
    assert statistics["independent_training_seeds"] == 2
    assert statistics["resampling_unit"] == "independent_training_seed_mean"


def test_bootstrap_is_deterministic_and_input_file_order_does_not_matter():
    records = [fixture_evaluation(seed, value=value) for seed, value in [(37, 3), (11, 1), (23, 2)]]
    first = aggregate_evaluations(records, bootstrap_samples=1000, bootstrap_seed=19)
    second = aggregate_evaluations(list(reversed(records)), bootstrap_samples=1000, bootstrap_seed=19)
    assert first == second
    assert first["training_seeds"] == [11, 23, 37]


def test_one_seed_has_no_estimated_between_seed_uncertainty():
    statistics = seed_statistics([7.0])
    assert statistics["mean"] == 7
    assert statistics["standard_deviation"] is None
    assert statistics["bootstrap_mean_95_percent_interval"] is None


def test_duplicate_training_seeds_are_rejected():
    record = fixture_evaluation(11)
    with pytest.raises(ValueError, match="Duplicate training seeds"):
        aggregate_evaluations([record, deepcopy(record)])


@pytest.mark.parametrize("field,value", [
    ("env", "mamujoco"), ("algorithm", "ippo"), ("map_name", "MMM2"),
    ("coordination", "causal"), ("total_steps", 20000000),
])
def test_mixed_tasks_algorithms_and_training_configs_are_rejected(field, value):
    records = [fixture_evaluation(11), fixture_evaluation(23)]
    records[1]["config"][field] = value
    with pytest.raises(ValueError, match="Mismatched"):
        aggregate_evaluations(records)


def test_mixed_completed_budgets_are_rejected():
    with pytest.raises(ValueError, match="training_steps"):
        aggregate_evaluations([fixture_evaluation(11), fixture_evaluation(23, steps=1000)])


def test_mixed_episode_counts_are_rejected():
    with pytest.raises(ValueError, match="evaluation_episodes"):
        aggregate_evaluations([fixture_evaluation(11), fixture_evaluation(23, episodes=20)])


@pytest.mark.parametrize("section", ["environment", "versions"])
def test_mixed_environment_settings_or_versions_are_rejected(section):
    records = [fixture_evaluation(11), fixture_evaluation(23)]
    records[1]["manifest"][section]["torch" if section == "versions" else "different"] = "different-fixture"
    with pytest.raises(ValueError, match="Mismatched"):
        aggregate_evaluations(records)


@pytest.mark.parametrize("missing", ["config", "manifest", "environment", "versions", "evaluation_protocol"])
def test_paper_comparison_requires_complete_metadata(targets, missing):
    record = fixture_evaluation(11)
    if missing in {"environment", "versions"}:
        record["manifest"].pop(missing)
    else:
        record.pop(missing)
    with pytest.raises(ValueError, match="requires|metadata"):
        aggregate_evaluations([record], target_id="3s5z", targets=targets)


def test_old_metadata_free_results_can_be_summarized_without_paper_comparison():
    record = fixture_evaluation(101)
    for key in ["config", "manifest", "evaluation_protocol"]:
        record.pop(key)
    report = aggregate_evaluations([record])
    assert report["comparison"] is None
    assert report["paper_reproduction_pass"] is False


def test_development_seeds_and_short_runs_cannot_pass_release_gate(targets):
    records = [fixture_evaluation(seed, steps=1000) for seed in [101, 103, 107]]
    comparison = aggregate_evaluations(records, target_id="3s5z", targets=targets)["comparison"]
    assert comparison["mean_exceeds_reference_point"]
    assert not comparison["release_reference_gate"]
    assert "registered_final_training_seeds" in comparison["failed_release_reference_checks"]
    assert "matched_environment_steps" in comparison["failed_release_reference_checks"]
    assert "registered_evaluation_episode_count" in comparison["failed_release_reference_checks"]


def test_smac_final_gate_is_reference_exceedance_not_full_paper_reproduction(targets):
    records = [fixture_evaluation(seed, episodes=1000) for seed in [11, 23, 37, 53, 71, 89]]
    report = aggregate_evaluations(records, target_id="3s5z", targets=targets)
    comparison = report["comparison"]
    assert comparison["release_reference_gate"]
    assert comparison["threshold"] == 0.975
    assert comparison["threshold_kind"] == "exact_published_point"
    assert comparison["scope"] == "published_point_under_declared_evaluation"
    assert "not an exact statistical reproduction" in comparison["interpretation"]
    assert not report["paper_reproduction_pass"] and not comparison["paper_reproduction_pass"]


def test_equal_published_score_is_not_improvement_due_to_float_rounding(targets):
    records = [fixture_evaluation(seed, episodes=1000, wins=975) for seed in [11, 23, 37, 53, 71, 89]]
    comparison = aggregate_evaluations(records, target_id="3s5z", targets=targets)["comparison"]
    assert not comparison["mean_exceeds_reference_point"]
    assert not comparison["release_reference_gate"]


def test_modern_mujoco_score_above_reference_never_claims_original_reproduction(targets):
    records = [fixture_evaluation(seed, env="mamujoco", episodes=100, value=10000)
               for seed in [11, 23, 37, 53, 71, 89]]
    report = aggregate_evaluations(records, target_id="HalfCheetah_2x3", targets=targets)
    comparison = report["comparison"]
    assert comparison["threshold"] == 3800
    assert comparison["source_kind"] == "visual_estimate_from_manuscript_raster_figure"
    assert comparison["scope"] == "modern_adaptation_reference_only"
    assert comparison["release_reference_gate"]
    assert not comparison["paper_reproduction_pass"] and not report["paper_reproduction_pass"]


@pytest.mark.parametrize("declaration", ["config", "training", "evaluation", "nested", "track"])
def test_computed_ant_forces_keep_reference_summary_but_block_certification(targets, declaration):
    records = [fixture_evaluation(seed, env="mamujoco", episodes=100, value=10000)
               for seed in [11, 23, 37, 53, 71, 89]]
    for record in records:
        record["config"].update(scenario="Ant", agent_conf="4x2", mamujoco_backend="legacy")
        for metadata in (record["manifest"]["environment"], record["evaluation_environment"]):
            metadata.update(scenario="Ant", agent_conf="4x2",
                            benchmark_track="historical_gym_v2_mujoco210_compatibility")
        if declaration == "config":
            record["config"]["ant_contact_forces"] = "computed"
        elif declaration in {"training", "evaluation"}:
            metadata = (record["manifest"]["environment"] if declaration == "training"
                        else record["evaluation_environment"])
            metadata["ant_contact_forces"] = "computed"
        else:
            for metadata in (record["manifest"]["environment"], record["evaluation_environment"]):
                if declaration == "nested":
                    metadata["contact_force_computation"] = {"mode": "computed"}
                else:
                    metadata["benchmark_track"] = "historical_gym_v2_ant_computed_forces_adaptation"
    report = aggregate_evaluations(records, target_id="Ant_4x2", targets=targets)
    comparison = report["comparison"]
    assert comparison["scope"] == "computed_ant_contact_forces_adaptation_reference_only"
    assert comparison["mean_exceeds_reference_point"]
    assert not comparison["release_reference_gate"]
    assert "no_unverified_contact_force_override" in comparison["failed_release_reference_checks"]
    assert not report["paper_reproduction_pass"]


def test_final_test_episode_seeds_must_not_reuse_training_diagnostics(targets):
    records = [fixture_evaluation(seed, episodes=1000) for seed in [11, 23, 37, 53, 71, 89]]
    for record in records:
        record["config"]["eval_seed"] = 500001
    comparison = aggregate_evaluations(records, target_id="3s5z", targets=targets)["comparison"]
    assert not comparison["release_reference_gate"]
    assert not comparison["release_reference_checks"]["final_episode_seeds_disjoint_from_diagnostics"]


@pytest.mark.parametrize("damage", ["mean", "nonfinite", "wins", "duplicates", "count"])
def test_raw_episode_evidence_is_validated(damage):
    record = fixture_evaluation(11)
    if damage == "mean":
        record["return_mean"] += 1
    elif damage == "nonfinite":
        record["results"][0]["return"] = float("nan")
    elif damage == "wins":
        record["win_rate"] = 0
    elif damage == "duplicates":
        record["results"][1]["seed"] = record["results"][0]["seed"]
    else:
        record["episodes"] += 1
    with pytest.raises(ValueError):
        aggregate_evaluations([record])


def test_target_must_match_recorded_task(targets):
    with pytest.raises(ValueError, match="does not match"):
        aggregate_evaluations([fixture_evaluation(11)], target_id="MMM2", targets=targets)


def test_cli_writes_json_and_refuses_overwrite_using_only_test_fixtures(tmp_path, monkeypatch):
    source = tmp_path / "manufactured-fixture.json"
    output = tmp_path / "fixture-aggregation.json"
    source.write_text(json.dumps(fixture_evaluation(101)))
    monkeypatch.setattr(sys, "argv", ["report", "--inputs", str(source), "--output", str(output)])
    main()
    report = json.loads(output.read_text())
    assert report["training_seeds"] == [101]
    assert report["paper_reproduction_pass"] is False
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_different_training_source_snapshots_cannot_be_aggregated():
    records = [fixture_evaluation(11), fixture_evaluation(23)]
    records[1]["manifest"]["source_sha256"]["dms/train.py"] = "c" * 64
    with pytest.raises(ValueError, match="training source snapshots"):
        aggregate_evaluations(records)


def test_runtime_job_device_and_host_differences_do_not_change_source_identity():
    records = [fixture_evaluation(11), fixture_evaluation(23)]
    records[0]["manifest"]["runtime"] = {"job_id": "fixture-job-a", "host": "fixture-cpu", "device": "cpu"}
    records[1]["manifest"]["runtime"] = {"job_id": "fixture-job-b", "host": "fixture-gpu", "device": "cuda"}
    records[0]["config"]["device"] = "cpu"
    records[1]["config"]["device"] = "cuda"
    report = aggregate_evaluations(records)
    assert report["provenance"]["training_sources_complete"]
    assert report["provenance"]["evaluation_sources_match_training"]
    assert not report["provenance"]["warnings"]


@pytest.mark.parametrize("damage", ["missing_training", "missing_evaluation", "mismatch", "invalid", "missing_file"])
def test_missing_or_changed_evaluation_provenance_blocks_gate_but_preserves_summary(targets, damage):
    records = [fixture_evaluation(seed, episodes=1000) for seed in [11, 23, 37, 53, 71, 89]]
    record = records[0]
    if damage == "missing_training":
        record["manifest"].pop("source_sha256")
    elif damage == "missing_evaluation":
        record.pop("evaluation_source_sha256")
    elif damage == "mismatch":
        record["evaluation_source_sha256"]["dms/train.py"] = "c" * 64
    elif damage == "invalid":
        record["evaluation_source_sha256"] = {"dms/train.py": "not-a-sha256"}
    else:
        record["evaluation_source_sha256"].pop("dms/evaluate.py")
    report = aggregate_evaluations(records, target_id="3s5z", targets=targets)
    assert report["metrics"]["win_rate"]["mean"] == 1
    assert report["provenance"]["warnings"]
    assert not report["comparison"]["release_reference_gate"]
    assert not report["comparison"]["release_reference_checks"]["evaluation_source_matches_training"]


def test_legacy_results_without_hashes_remain_available_with_explicit_warnings():
    records = [fixture_evaluation(101), fixture_evaluation(103)]
    for record in records:
        record["manifest"].pop("source_sha256")
        record.pop("evaluation_source_sha256")
    report = aggregate_evaluations(records)
    assert len(report["provenance"]["warnings"]) == 4
    assert not report["provenance"]["training_sources_complete"]
    assert not report["provenance"]["evaluation_sources_match_training"]


def test_resolved_map_identity_is_portable_without_mutating_or_discarding_raw_provenance(targets):
    records = [fixture_evaluation(seed, episodes=1000) for seed in [11, 23, 37, 53, 71, 89]]
    for index, record in enumerate(records):
        record["manifest"]["environment"]["map_asset"]["path"] = f"/training-machine-{index}/3s5z.SC2Map"
        record["evaluation_environment"]["map_asset"]["path"] = f"/evaluation-machine-{index}/3s5z.SC2Map"
    original = deepcopy(records)
    report = aggregate_evaluations(records, target_id="3s5z", targets=targets)
    assert records == original
    assert report["comparison"]["release_reference_gate"]
    assert "path" not in report["experiment"]["environment_metadata"]["map_asset"]
    for record, raw in zip(records, report["recorded_provenance"]):
        assert raw["manifest"] == record["manifest"]
        assert raw["evaluation_environment"] == record["evaluation_environment"]
        assert raw["evaluation_versions"] == record["evaluation_versions"]
    report["recorded_provenance"][0]["manifest"]["environment"]["map_asset"]["path"] = "changed-output-copy"
    assert records == original


@pytest.mark.parametrize("status,digest", [("unresolved", "a" * 64), ("resolved", None), ("resolved", "not-a-hash")])
def test_map_installation_path_is_strict_without_resolved_valid_content_identity(status, digest):
    records = [fixture_evaluation(11), fixture_evaluation(23)]
    for index, record in enumerate(records):
        record["manifest"]["environment"]["map_asset"].update(
            status=status, sha256=digest, path=f"/fixture-{index}/3s5z.SC2Map")
    with pytest.raises(ValueError, match="environment_metadata"):
        aggregate_evaluations(records)


@pytest.mark.parametrize("section,field,value", [
    ("map_asset", "sha256", "b" * 64),
    ("map_asset", "relative_path", "OtherMaps/3s5z.SC2Map"),
    ("runtime_selection", "build_version", 75800),
    ("runtime_selection", "data_version", "different-data-version"),
    ("process_lifetime", "restart_at_episode_boundary_after_game_loops", 400000),
    ("process_lifetime", "seed_schedule", "different-seed-policy"),
    (None, "xml_sha256", "b" * 64),
    (None, "mujoco_engine_version", "different-engine"),
    (None, "other_path", "/must/not/be/ignored"),
    (None, "runtime", {"unknown_environment_setting": True}),
])
def test_only_verified_map_location_is_normalized(section, field, value):
    records = [fixture_evaluation(11), fixture_evaluation(23)]
    environment = records[1]["manifest"]["environment"]
    (environment if section is None else environment[section])[field] = value
    with pytest.raises(ValueError, match="environment_metadata"):
        aggregate_evaluations(records)


@pytest.mark.parametrize("family,backend,unused", [
    ("smac", "modern", ["mujoco", "gymnasium", "gymnasium-robotics", "gym", "mujoco-py"]),
    ("mamujoco", "modern", ["smac", "pysc2", "s2clientprotocol", "protobuf", "gym", "mujoco-py"]),
    ("mamujoco", "legacy", ["smac", "pysc2", "s2clientprotocol", "protobuf", "gymnasium", "gymnasium-robotics", "mujoco"]),
])
def test_unused_optional_package_versions_do_not_define_the_experiment(family, backend, unused):
    records = [fixture_evaluation(11, env=family), fixture_evaluation(23, env=family)]
    for record in records:
        record["config"]["mamujoco_backend"] = backend
    for package in unused:
        records[1]["manifest"]["versions"][package] = "unused-training-version"
        records[0]["evaluation_versions"][package] = "unused-evaluation-version"
    report = aggregate_evaluations(records)
    assert report["provenance"]["evaluation_dependency_versions_match_training"]
    assert not report["provenance"]["warnings"]
    assert report["recorded_provenance"][1]["manifest"]["versions"][unused[0]] == "unused-training-version"


@pytest.mark.parametrize("family,backend,package", [
    ("smac", "modern", "torch"), ("smac", "modern", "numpy"),
    ("smac", "modern", "pysc2"), ("smac", "modern", "smac"),
    ("smac", "modern", "s2clientprotocol"), ("smac", "modern", "protobuf"),
    ("mamujoco", "modern", "gymnasium"), ("mamujoco", "modern", "gymnasium-robotics"),
    ("mamujoco", "modern", "mujoco"), ("mamujoco", "legacy", "gym"),
    ("mamujoco", "legacy", "mujoco-py"),
])
def test_used_package_changes_cannot_be_aggregated(family, backend, package):
    records = [fixture_evaluation(11, env=family), fixture_evaluation(23, env=family)]
    for record in records:
        record["config"]["mamujoco_backend"] = backend
    records[1]["manifest"]["versions"][package] = "different-used-version"
    with pytest.raises(ValueError, match="dependency_versions"):
        aggregate_evaluations(records)


@pytest.mark.parametrize("damage,failed_check", [
    ("missing_environment", "evaluation_environment_matches_training"),
    ("missing_versions", "evaluation_dependency_versions_match_training"),
    ("missing_used_version", "evaluation_dependency_versions_match_training"),
    ("map_hash", "evaluation_environment_matches_training"),
    ("build", "evaluation_environment_matches_training"),
    ("lifetime", "evaluation_environment_matches_training"),
    ("used_package", "evaluation_dependency_versions_match_training"),
])
def test_actual_evaluation_provenance_must_match_training_to_pass_gate(targets, damage, failed_check):
    records = [fixture_evaluation(seed, episodes=1000) for seed in [11, 23, 37, 53, 71, 89]]
    record = records[0]
    if damage == "missing_environment":
        record.pop("evaluation_environment")
    elif damage == "missing_versions":
        record.pop("evaluation_versions")
    elif damage == "missing_used_version":
        record["evaluation_versions"].pop("numpy")
    elif damage == "map_hash":
        record["evaluation_environment"]["map_asset"]["sha256"] = "b" * 64
    elif damage == "build":
        record["evaluation_environment"]["runtime_selection"]["build_version"] += 1
    elif damage == "lifetime":
        record["evaluation_environment"]["process_lifetime"]["restart_at_episode_boundary_after_game_loops"] = 400000
    else:
        record["evaluation_versions"]["protobuf"] = "different-used-version"
    report = aggregate_evaluations(records, target_id="3s5z", targets=targets)
    assert report["metrics"]["win_rate"]["mean"] == 1
    assert report["provenance"]["evaluation_sources_match_training"]
    assert report["provenance"]["warnings"]
    assert not report["comparison"]["release_reference_gate"]
    assert failed_check in report["comparison"]["failed_release_reference_checks"]


def test_runtime_counters_are_recorded_separately_and_do_not_define_environment_identity(targets):
    records = [fixture_evaluation(seed, episodes=1000) for seed in [11, 23, 37, 53, 71, 89]]
    for index, record in enumerate(records):
        record["manifest"]["environment_runtime"] = [{"scheduled_restarts": index, "game_loop": 1000 * index}]
        record["evaluation_runtime"] = {"host": f"fixture-{index}", "device": "cpu",
                                        "final_environment": {"game_loop": 100 + index, "process_seed": 20001 + index}}
    report = aggregate_evaluations(records, target_id="3s5z", targets=targets)
    assert report["comparison"]["release_reference_gate"]
    assert not report["provenance"]["warnings"]


def test_backend_package_fallback_supports_old_manifests_but_conflicts_block_gate(targets):
    records = [fixture_evaluation(seed, episodes=1000) for seed in [11, 23, 37, 53, 71, 89]]
    for record in records:
        versions = record["manifest"]["versions"]
        backend = {name: versions.pop(name) for name in ["s2clientprotocol", "protobuf"]}
        record["manifest"]["environment"]["packages"] = deepcopy(backend)
        record["evaluation_environment"]["packages"] = deepcopy(backend)
    assert aggregate_evaluations(records, target_id="3s5z", targets=targets)["comparison"]["release_reference_gate"]
    records[0]["evaluation_versions"]["protobuf"] = "conflicts-with-recorded-backend"
    report = aggregate_evaluations(records, target_id="3s5z", targets=targets)
    assert not report["comparison"]["release_reference_gate"]
    assert any("conflicting evaluation dependency versions" in warning for warning in report["provenance"]["warnings"])
