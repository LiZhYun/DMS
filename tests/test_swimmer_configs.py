import json
from pathlib import Path

import pytest

from dms.swimmer_configs import generate_configs, main
from dms.train import Config


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/development/swimmer10x2-dms-causal-pilot.json"
TARGETS = ROOT / "configs/paper_targets.json"


def generate(output, **kwargs):
    return generate_configs(output, template=TEMPLATE, targets=TARGETS, **kwargs)


def read_matrix(path):
    manifest = json.loads(path.read_text())
    configs = {}
    for entry in manifest["experiments"]:
        value = json.loads((path.parent / entry["config"]).read_text())
        Config(**value).validate()
        configs[entry["id"]] = value
    return manifest, configs


def test_supplemental_matrix_has_seven_full_budget_configs_and_never_runs(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("config generation must not construct environments or run training")

    monkeypatch.setattr("dms.train.create_env", forbidden)
    monkeypatch.setattr("dms.train.run", forbidden)
    manifest, configs = read_matrix(generate(tmp_path / "k1"))
    assert len(configs) == 7
    assert manifest["primary_benchmark_gate"] is False
    assert manifest["status"] == "unverified_provisional_configurations_not_results"
    assert manifest["agent_obsk"] == 1 and manifest["observed_peer_controls"] is True
    assert manifest["reserved_final_training_seeds"] == [11,23,37,53,71,89]
    assert manifest["development_seeds"] == [101,103,107]
    for config in configs.values():
        assert config["total_steps"] == 10_000_000 and config["seed"] == 101
        assert config["env"] == "mamujoco" and config["mamujoco_backend"] == "legacy"
        assert config["scenario"] == "ManyAgentSwimmer" and config["agent_conf"] == "10x2"
        assert config["coordination"] == "causal" and config["filter_history"] is True
        assert config["decoder_history"] is False and config["kl_beta"] == .1
        assert config["history_length"] == 32  # latent-only decoder does not remove past sensing
        assert config["eval_every"] == 1_000_000 and config["eval_episodes"] == 10


def test_studies_share_reference_and_change_only_the_declared_parameters(tmp_path):
    manifest, configs = read_matrix(generate(tmp_path))
    reference = configs["reference"]
    differences = {identity:{key for key,value in config.items() if value != reference[key]} for identity,config in configs.items()}
    assert differences["ippo"] == {"algorithm", "output"}
    for identity in ("delta-02","delta-08","delta-04-prose"):
        assert differences[identity] == {"coordination_threshold", "output"}
    for identity in ("longshort-k100-l6","longshort-k1e6-l2"):
        assert differences[identity] == {"intention_threshold", "short_history", "output"}
    studies = manifest["studies"]
    delta = studies["coordination_threshold"]
    assert [configs[key]["coordination_threshold"] for key in delta["plotted_experiment_ids"]] == [.2,.6,.8]
    assert delta["prose_only_experiment_ids"] == ["delta-04-prose"]
    assert "delta-04-prose" not in delta["plotted_experiment_ids"]
    longshort = studies["long_short_intention"]
    assert [(configs[key]["intention_threshold"], configs[key]["short_history"]) for key in longshort["plotted_experiment_ids"]] == [(100,6),(10_000,4),(1_000_000,2)]
    assert set(delta["plotted_experiment_ids"]) & set(longshort["plotted_experiment_ids"]) == {"reference"}


def test_explicit_k0_is_a_separate_no_peer_sensor_sensitivity(tmp_path):
    _, primary = read_matrix(generate(tmp_path / "k1"))
    manifest, sensitivity = read_matrix(generate(tmp_path / "k0", agent_obsk=0))
    assert manifest["observation_track"] == "explicit_paper_table_k0_sensitivity"
    assert manifest["observed_peer_controls"] is False
    for identity, config in sensitivity.items():
        assert config["agent_obsk"] == 0
        assert {key for key,value in config.items() if value != primary[identity][key]} == {"agent_obsk","output"}
    assert {config["output"] for config in sensitivity.values()}.isdisjoint(config["output"] for config in primary.values())


@pytest.mark.parametrize("filename", ["manifest.json", "swimmer10x2-k1-reference.json"])
def test_existing_output_is_preserved_without_partial_generation(tmp_path, filename):
    existing = tmp_path / filename
    existing.write_text("preserve this\n")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        generate(tmp_path)
    assert existing.read_text() == "preserve this\n"
    assert list(tmp_path.iterdir()) == [existing]


@pytest.mark.parametrize("radius", [2, -1, True])
def test_invalid_sensing_track_fails_before_writing(tmp_path, radius):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="agent_obsk"):
        generate(output, agent_obsk=radius)
    assert not output.exists()


def test_soft_template_is_rejected_instead_of_running_ineffective_delta_sweep(tmp_path):
    template = json.loads(TEMPLATE.read_text())
    template["coordination"] = "soft"
    path = tmp_path / "soft.json"
    path.write_text(json.dumps(template))
    output = tmp_path / "generated"
    with pytest.raises(ValueError, match="coordination"):
        generate_configs(output, template=path, targets=TARGETS)
    assert not output.exists()


def test_cli_writes_only_explicit_supplemental_directory(tmp_path, capsys):
    main(["--template",str(TEMPLATE),"--targets",str(TARGETS),"--output-dir",str(tmp_path),"--agent-obsk","0"])
    assert "7 supplemental configurations" in capsys.readouterr().out
    manifest, _ = read_matrix(tmp_path / "manifest.json")
    assert manifest["agent_obsk"] == 0


def test_committed_pilot_pair_is_matched_and_reaches_past_causal_warmup():
    ippo = json.loads((ROOT / "configs/development/swimmer10x2-ippo-pilot.json").read_text())
    dms = json.loads(TEMPLATE.read_text())
    Config(**ippo).validate()
    Config(**dms).validate()
    assert {key for key in ippo if ippo[key] != dms[key]} == {"algorithm", "output"}
    assert dms["total_steps"] == 500_000 and dms["eval_every"] == 100_000
    assert dms["eval_episodes"] == 5
    assert dms["total_steps"] > 1000 * (dms["warmup_episodes"]+dms["causal_warmup_episodes"])
