from collections import Counter, defaultdict
import json
from pathlib import Path

import pytest

from dms.benchmark_configs import generate_configs, main
from dms.train import Config


TARGETS = Path(__file__).resolve().parents[1] / "configs" / "paper_targets.json"


def _read_matrix(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    grouped = defaultdict(dict)
    for entry in manifest["experiments"]:
        values = json.loads((manifest_path.parent / entry["config"]).read_text())
        Config(**values).validate()
        grouped[entry["task"]][entry["variant"]] = values
    return manifest, grouped


def test_primary_matrix_has_all_twelve_tasks_with_matched_full_budgets(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("configuration generation must not construct environments or train")

    monkeypatch.setattr("dms.train.create_env", forbidden)
    monkeypatch.setattr("dms.train.run", forbidden)
    manifest_path = generate_configs(TARGETS, tmp_path / "generated")
    manifest, tasks = _read_matrix(manifest_path)
    assert len(manifest["experiments"]) == 24
    assert len(tasks) == 12
    assert Counter(entry["environment"] for entry in manifest["experiments"]) == {"mamujoco": 18, "smac": 6}
    expected = json.loads(TARGETS.read_text())
    assert set(tasks) == {task["id"] for family in ("mamujoco", "smac") for task in expected[family]}
    assert manifest["training_seeds"] == expected["reproduction_evaluation_policy"]["training_seeds"]
    assert manifest["development_seeds"] == expected["reproduction_evaluation_policy"]["development_seeds"]
    for variants in tasks.values():
        assert set(variants) == {"ippo", "dms-soft"}
        ippo, dms = variants["ippo"], variants["dms-soft"]
        assert {key for key in ippo if ippo[key] != dms[key]} == {"algorithm", "output"}
        for values in variants.values():
            assert values["total_steps"] == 10_000_000
            assert values["num_envs"] == 4
            assert values["rollout_steps"] == (1000 if values["env"] == "mamujoco" else 160)
            assert values["seed"] == 101
            assert values["agent_obsk"] == 1
            assert values["warmup_episodes"] == 100
            assert values["filter_history"] is True
            assert "training_seeds" not in values


def test_ablation_matrix_changes_only_declared_variant_fields(tmp_path):
    manifest, tasks = _read_matrix(generate_configs(TARGETS, tmp_path / "ablations", steps=20_000, agent_obsk=0, include_ablations=True))
    assert len(manifest["experiments"]) == 60
    assert manifest["observation_track"] == "paper_table_k0_no_observed_peers"
    for variants in tasks.values():
        base = variants["dms-soft"]
        assert variants["dms-causal"]["coordination"] == "causal"
        assert variants["dms-all"]["coordination"] == "all"
        for variant, values in variants.items():
            assert values["total_steps"] == 20_000
            assert values["agent_obsk"] == 0
            assert values["filter_history"] is (variant != "dms-no-history-filter")
            differences = {key for key in values if values[key] != base[key]}
            assert differences <= {"algorithm", "coordination", "filter_history", "output"}
        assert {key for key in variants["dms-no-history-filter"] if variants["dms-no-history-filter"][key] != base[key]} == {"filter_history", "output"}


@pytest.mark.parametrize("filename", ["manifest.json", "mamujoco-ant-2x4-ippo.json"])
def test_existing_files_are_preserved_and_no_partial_matrix_is_written(tmp_path, filename):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    existing = output_dir / filename
    existing.write_text("user-owned contents\n")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        generate_configs(TARGETS, output_dir)
    assert existing.read_text() == "user-owned contents\n"
    assert list(output_dir.iterdir()) == [existing]


def test_invalid_budget_fails_config_validation_before_creating_output(tmp_path):
    output_dir = tmp_path / "invalid"
    with pytest.raises(ValueError, match="divisible"):
        generate_configs(TARGETS, output_dir, steps=10_000_001)
    assert not output_dir.exists()


def test_cli_writes_only_to_requested_temporary_directory(tmp_path, capsys):
    output_dir = tmp_path / "cli"
    main(["--targets", str(TARGETS), "--output-dir", str(output_dir), "--steps", "4000"])
    assert "Generated 24 configurations" in capsys.readouterr().out
    manifest, tasks = _read_matrix(output_dir / "manifest.json")
    assert manifest["total_environment_steps"] == 4000
    assert all(values["total_steps"] == 4000 for variants in tasks.values() for values in variants.values())
