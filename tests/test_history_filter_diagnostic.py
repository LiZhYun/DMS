import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.distributions import Independent, Normal, kl_divergence

from dms.envs import CoordinationToy
from dms.intention import DiagonalGaussian, IntentionModel, gaussian_kl
from dms.train import Config, Learner


spec = importlib.util.spec_from_file_location(
    "history_filter_probe", Path(__file__).resolve().parents[1]/"scripts/diagnose_history_filter.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_probe_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixed_model(monkeypatch):
    model = IntentionModel(3, 2, hidden_dim=8, latent_dim=2,
                           observation_mask=torch.tensor([[1,1],[1,0],[0,0]],dtype=torch.bool))
    long = DiagonalGaussian(torch.zeros(2,3,2), torch.zeros(2,3,2))
    short = DiagonalGaussian(torch.ones(2,3,2), torch.full((2,3,2), np.log(2.0)))
    monkeypatch.setattr(model, "prior", lambda past, visible: long if past.shape[1] == 8 else short)
    past = torch.zeros(2,8,3,2)
    visible = torch.zeros(2,8,3,dtype=torch.bool)
    visible[0,:,0] = True
    visible[0,-4:,1] = True
    visible[1,:2,1] = True  # older-only history is not eligible for suffix rejection
    return model,past,visible,long,short


def test_kl_direction_latent_sum_and_recent_visibility_match_production(monkeypatch):
    model,past,visible,long,short = fixed_model(monkeypatch)
    saved = visible.clone()
    rows = probe.inspect_filter(model,past,visible,0,settings=[(.1,4)])
    peer = next(row for row in rows if row["peer"] == 1)
    expected = kl_divergence(Independent(Normal(long.mean,long.log_std.exp()),1),
                             Independent(Normal(short.mean,short.log_std.exp()),1))[0,1].item()
    reversed_kl = gaussian_kl(short,long)[0,1].item()
    assert peer["raw"]["kl"] == pytest.approx([expected])
    assert expected != pytest.approx(reversed_kl)
    assert peer["kl_mean"] != pytest.approx(expected/2)  # no mean over latent dimensions
    assert peer["raw"]["sample_indices"] == [0]
    assert peer["eligible_count"] == peer["rejected_count"] == 1
    assert peer["removed_visible_action_entries"] == 4
    assert rows[0]["self"] is True and rows[2]["eligible_count"] == 0
    assert torch.equal(visible,saved)


def test_threshold_boundary_is_strict_and_old_visible_actions_remain(monkeypatch):
    model,past,visible,long,short = fixed_model(monkeypatch)
    cutoff = gaussian_kl(long,short)[0,1].item()
    rows = probe.inspect_filter(model,past,visible,0,settings=[(cutoff,4)])
    assert all(row["rejected_count"] == 0 for row in rows)
    assert all(row["removed_visible_action_entries"] == 0 for row in rows)


def test_unknown_history_has_empty_statistics_not_artificial_zero_kl():
    model = IntentionModel(2,1,hidden_dim=8,latent_dim=2)
    rows = probe.inspect_filter(model,torch.zeros(2,8,2,1),torch.zeros(2,8,2,dtype=torch.bool),0)
    assert len(rows) == 6
    for row in rows:
        assert row["eligible_count"] == row["rejected_count"] == 0
        assert row["rejection_fraction"] is None and row["kl_mean"] is None
        assert row["kl_quantiles"] is None and row["raw"]["kl"] == []


def test_hidden_components_and_actions_cannot_change_measurements(monkeypatch):
    model = IntentionModel(2,2,hidden_dim=8,latent_dim=2,
                           observation_mask=torch.tensor([[1,1],[1,0]],dtype=torch.bool))
    monkeypatch.setattr(model,"posterior",lambda *args: pytest.fail("future posterior was used"))
    past = torch.randn(3,8,2,2)
    visible = torch.ones(3,8,2,dtype=torch.bool)
    visible[:,1::2,1] = False
    before = probe.inspect_filter(model,past,visible,0)
    changed = past.clone()
    changed[:,1::2,1] = 1e9
    changed[:,:,1,1] = -1e9
    assert probe.inspect_filter(model,changed,visible,0) == before


def test_invalid_visibility_and_incompatible_filter_are_rejected(monkeypatch):
    model,past,visible,_,_ = fixed_model(monkeypatch)
    bad = visible.clone(); bad[0,-1,2] = True
    with pytest.raises(ValueError,match="local component mask"):
        probe.inspect_filter(model,past,bad,0)
    monkeypatch.setattr(model,"filtered_history",lambda p,v,l,k: (v.clone(),torch.zeros(2,3,dtype=torch.bool)))
    with pytest.raises(RuntimeError,match="differ from"):
        probe.inspect_filter(model,past,visible,0,settings=[(0,4)])


def test_descriptive_quantiles_are_not_paper_thresholds_or_seed_confidence_intervals():
    result = probe.summarize([1,2,3,4],100)
    assert result["eligible_count"] == 4 and result["rejection_fraction"] == 0
    assert result["kl_quantiles"]["0.5"] == 2.5
    median = next(row for row in result["development_quantile_thresholds"] if row["quantile"] == .5)
    assert median["fraction_strictly_above"] == .5
    with pytest.raises(ValueError,match="finite"):
        probe.summarize([float("nan")],100)


def toy_learner():
    config = Config(hidden_dim=8,latent_dim=2,history_length=8,prediction_horizon=2,
                    num_envs=1,filter_history=True,warmup_episodes=0)
    learner = Learner(config,CoordinationToy(horizon=3))
    learner.completed_episodes = 5
    return learner


def test_same_prefix_probe_ignores_future_fields_and_separates_self(monkeypatch):
    learner = toy_learner()
    env = CoordinationToy(horizon=3)
    before = probe.helpers.learner_state_hash(learner)
    data, _ = probe.helpers.collect_fresh(learner,[env],8,400001)
    data["future"] = object(); data["future_visible"] = object()
    for model in learner.intentions:
        monkeypatch.setattr(model,"posterior",lambda *args: pytest.fail("future posterior used"))
    results = probe.diagnose(learner,data)
    assert len(results["settings"]) == 3
    for setting in results["settings"]:
        assert setting["paper_joint_setting"]
        peers = [row for row in results["per_observer_peer"] if not row["self"] and
                 row["threshold"] == setting["threshold"] and row["short_length"] == setting["short_length"]]
        assert setting["peers"]["eligible_count"] == sum(row["eligible_count"] for row in peers)
    assert probe.helpers.learner_state_hash(learner) == before
    assert learner.completed_episodes == 5


def test_source_fingerprint_includes_packaged_xml(tmp_path,monkeypatch):
    package = tmp_path/"dms"; package.mkdir()
    (package/"__init__.py").write_text("")
    assets = package/"assets"; assets.mkdir()
    xml = assets/"robot.xml"; xml.write_text("<mujoco/>")
    monkeypatch.setattr(probe.dms,"__file__",str(package/"__init__.py"))
    before = probe.source_hashes()
    xml.write_text("<mujoco model='changed'/>")
    after = probe.source_hashes()
    assert before["dms/assets/robot.xml"] != after["dms/assets/robot.xml"]
    assert "scripts/diagnose_history_filter.py" in before and "scripts/diagnose_intentions.py" in before


@pytest.fixture
def cli(tmp_path,monkeypatch):
    learner = toy_learner()
    checkpoint = tmp_path/"frozen.pt"
    learner.save(checkpoint,500_000,5)
    checkpoint.chmod(0o444)
    closed = []
    class Toy(CoordinationToy):
        def close(self): closed.append(True)
    monkeypatch.setenv("SLURM_JOB_ID","fixture")
    monkeypatch.setattr(probe,"create_env",lambda config: Toy(horizon=3))
    monkeypatch.setattr(probe,"source_hashes",lambda: {"dms/assets/fixture.xml":"stable"})
    output = tmp_path/"filter.json"
    args = ["--checkpoint",str(checkpoint),"--output",str(output),"--steps","8","--num-envs","1"]
    return checkpoint,output,args,closed


def test_cli_fresh_trajectories_preserve_checkpoint_state_and_record_scope(cli):
    checkpoint,output,args,closed = cli
    before = checkpoint.read_bytes()
    probe.main(args)
    result = json.loads(output.read_text())
    assert result["out_of_training_transitions"] == 8
    assert result["parameter_updates"] == 0 and result["training_steps"] == 500_000
    assert result["checkpoint_unchanged"] and result["source_and_xml_unchanged"] and result["model_state_unchanged"]
    assert result["checkpoint_read_only"] and checkpoint.read_bytes() == before
    assert result["collection"]["completed_diagnostic_episodes"] == 2
    assert result["filter_active_in_collected_policy"] is True
    assert len(result["settings"]) == 3 and closed == [True]
    with pytest.raises(FileExistsError):
        probe.main(args)


@pytest.mark.parametrize("mutation",["model","source","checkpoint","permissions"])
def test_mutations_fail_without_publishing_output(cli,monkeypatch,mutation):
    checkpoint,output,args,closed = cli
    original = probe.helpers.collect_fresh
    def changed(learner,*arguments):
        result = original(learner,*arguments)
        if mutation == "model":
            learner.agents[0].obs_moments.mean.add_(1)
        elif mutation == "checkpoint":
            checkpoint.chmod(0o644)
            checkpoint.write_bytes(b"replaced")
        elif mutation == "permissions":
            checkpoint.chmod(0o644)
        else:
            monkeypatch.setattr(probe,"source_hashes",lambda: {"dms/assets/fixture.xml":"changed"})
        return result
    monkeypatch.setattr(probe.helpers,"collect_fresh",changed)
    with pytest.raises(RuntimeError,match="changed"):
        probe.main(args)
    assert not output.exists() and closed == [True]


def test_mutable_checkpoint_is_rejected(cli):
    checkpoint,output,args,closed = cli
    checkpoint.chmod(0o644)
    with pytest.raises(SystemExit):
        probe.main(args)
    assert not output.exists() and not closed


def test_cli_requires_allocation(tmp_path,monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID",raising=False)
    with pytest.raises(SystemExit):
        probe.main(["--checkpoint",str(tmp_path/"absent.pt"),"--output",str(tmp_path/"out.json")])
