import importlib.util
import json
from pathlib import Path

import pytest
import torch

from dms.envs import CoordinationToy
from dms.train import Config, Learner


spec = importlib.util.spec_from_file_location(
    'evaluate_interventions', Path(__file__).resolve().parents[1]/'scripts/evaluate_interventions.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def inputs(value):
    obs = torch.full((1, 2, 3), value)
    context = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) + value
    gates = torch.tensor([[[False, True], [True, False]]])
    available = torch.tensor([[[True, False], [False, True]]])
    return obs, context, gates, available


@pytest.mark.parametrize('mode', probe.MODES)
def test_interventions_preserve_self_query_observations_masks_and_original_inputs(mode):
    change = probe.PeerIntervention(mode)
    change.apply(*inputs(1.0))
    current = inputs(20.0)
    saved = [x.clone() for x in current]
    result = change.apply(*current)
    assert result[0] is current[0] and result[3] is current[3]
    for observer in range(2):
        assert torch.equal(result[1][:, observer, observer], current[1][:, observer, observer])
    assert all(torch.equal(a, b) for a, b in zip(current, saved))
    if mode == 'hold':
        assert torch.equal(result[1][0, 0, 1], inputs(1.0)[1][0, 0, 1])
        assert torch.equal(result[2], current[2])
    elif mode == 'masked':
        assert not result[2].any()
        assert torch.equal(result[1], current[1])
    else:
        assert all(torch.equal(a, b) for a, b in zip(current, result))


def test_hold_captures_only_first_eligible_value_and_resets_between_episodes():
    change = probe.PeerIntervention('hold')
    hidden = list(inputs(100.0)); hidden[2] = torch.zeros_like(hidden[2])
    change.apply(*hidden)
    assert change.captured_peer_entries == 0
    first = inputs(2.0)
    change.apply(*first)
    assert change.captured_peer_entries == 2
    hidden[1].fill_(9999)
    invisible_result = change.apply(*hidden)
    assert not invisible_result[2].any()
    assert torch.equal(invisible_result[1], hidden[1])
    later = change.apply(*inputs(-7.0))
    assert torch.equal(later[1][0, 0, 1], first[1][0, 0, 1])
    assert change.captured_peer_entries == 2
    change.reset()
    next_episode = inputs(17.0)
    reset_result = change.apply(*next_episode)
    assert torch.equal(reset_result[1], next_episode[1])
    assert change.captured_peer_entries == 2


def learner():
    config = Config(hidden_dim=8, latent_dim=2, history_length=4,
                    prediction_horizon=2, filter_history=False, warmup_episodes=0)
    return Learner(config, CoordinationToy(horizon=3))


def test_complete_episodes_match_normal_evaluation_and_preserve_all_model_state(monkeypatch):
    import dms.train as train

    model = learner()
    model.completed_episodes = 19
    before = probe.state_hash(model)
    monkeypatch.setattr(train, 'create_env', lambda config: CoordinationToy(horizon=3))
    expected = train.evaluate(model, 2, 300001)
    for intention in model.intentions:
        monkeypatch.setattr(intention, 'posterior', lambda *a, **k: pytest.fail('future-target posterior used'))
    records, metadata = probe.evaluate_modes(model, [300001, 300002], env_factory=train.create_env)
    actual = [{k: row[k] for k in ('seed', 'return', 'length', 'battle_won')}
              for row in records if row['mode'] == 'normal']
    assert actual == expected
    for mode in probe.MODES:
        rows = [r for r in records if r['mode'] == mode]
        assert [r['seed'] for r in rows] == [300001, 300002]
        assert [r['length'] for r in rows] == [3, 3]
        assert all(r['eligible_peer_entries'] == 6 for r in rows)
        assert all(r['captured_peer_entries'] == (3 if mode == 'hold' else 0) for r in rows)
    assert set(metadata) == set(probe.MODES)
    assert probe.state_hash(model) == before and model.completed_episodes == 19


def test_state_mutation_is_detected(monkeypatch):
    model = learner()
    original = model.action

    def mutate(*args, **kwargs):
        model.agents[0].obs_moments.mean.add_(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, 'action', mutate)
    with pytest.raises(RuntimeError, match='changed model state'):
        probe.evaluate_modes(model, [300001], env_factory=lambda c: CoordinationToy(horizon=1))


def test_incomplete_episode_is_not_reported_and_environment_closes():
    closed = []

    class LongToy(CoordinationToy):
        def close(self):
            closed.append(True)

    with pytest.raises(RuntimeError, match='episode exceeded'):
        probe.evaluate_modes(learner(), [300001], max_episode_steps=2,
                             env_factory=lambda c: LongToy(horizon=3))
    assert closed == [True]


@pytest.mark.parametrize('setting,value', [('algorithm', 'ippo'), ('coordination', 'causal'),
                                         ('coordination', 'all'), ('warmup_episodes', 1)])
def test_unsupported_or_inactive_models_are_rejected(setting, value):
    model = learner()
    setattr(model.config, setting, value)
    with pytest.raises(ValueError):
        probe.evaluate_modes(model, [300001])


def test_cli_records_raw_episodes_source_and_normalizer_invariants(tmp_path, monkeypatch):
    model = learner()
    checkpoint = tmp_path/'checkpoint.pt'
    model.save(checkpoint, 500000, 500)
    monkeypatch.setenv('SLURM_JOB_ID', 'test')
    monkeypatch.setattr(probe, 'create_env', lambda c: CoordinationToy(horizon=3))
    output = tmp_path/'intervention.json'
    args = ['--checkpoint', str(checkpoint), '--output', str(output), '--episodes', '2']
    probe.main(args)
    result = json.loads(output.read_text())
    assert len(result['records']) == 6 and result['out_of_training_transitions'] == 18
    assert result['episode_seeds'] == [300001, 300002]
    assert result['source_unchanged_during_evaluation'] and result['model_state_unchanged']
    assert result['parameter_updates'] == 0 and result['training_episodes'] == 500
    assert result['diagnostic_source_sha256']['scripts/evaluate_interventions.py']
    assert result['versions']['torch']
    with pytest.raises(FileExistsError):
        probe.main(args)


def test_cli_requires_allocation(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(SystemExit):
        probe.main(['--checkpoint', str(tmp_path/'missing.pt'), '--output', str(tmp_path/'out.json')])
