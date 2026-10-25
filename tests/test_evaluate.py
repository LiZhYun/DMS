import json
import sys
from copy import deepcopy

import pytest

from dms.evaluate import main
from dms.train import Config, run


def test_checkpoint_evaluator_preserves_provenance_and_raw_episodes(tmp_path, monkeypatch):
    run_dir = tmp_path/'training'
    config = Config(output=str(run_dir),num_envs=1,total_steps=4,rollout_steps=4,
                    history_length=4,prediction_horizon=2,hidden_dim=8,latent_dim=2,
                    epochs=1,minibatch_size=4,aux_updates=1,aux_batch_size=2,decoder_history=False,
                    eval_every=4,eval_episodes=1,seed=7)
    run(config)
    output = tmp_path/'diagnostic-evaluation.json'
    monkeypatch.setattr(sys,'argv',['dms.evaluate','--checkpoint',str(run_dir/'latest.pt'),
                        '--episodes','2','--seed','20001','--output',str(output)])
    main()
    result = json.loads(output.read_text())
    assert result['training_steps'] == 4
    assert result['training_seed'] == 7
    assert [row['seed'] for row in result['results']] == [20001,20002]
    assert result['episodes'] == 2
    assert result['manifest']['completed_steps'] == 4
    assert result['manifest']['source_sha256'] == result['evaluation_source_sha256']
    assert result['return_mean'] == sum(row['return'] for row in result['results'])/2
    assert result['evaluation_environment'] == result['manifest']['environment']
    assert result['evaluation_versions']['torch'] == result['manifest']['versions']['torch']
    assert result['evaluation_versions']['numpy'] == result['manifest']['versions']['numpy']
    assert result['evaluation_runtime']['device'] == 'cpu'
    assert result['evaluation_runtime']['initial_environment'] == {}
    assert result['evaluation_runtime']['final_environment'] == {}


def test_evaluator_records_rollout_environment_instead_of_training_or_dimension_probe(tmp_path, monkeypatch):
    import dms.train as train_module
    from dms.report import aggregate_evaluations

    run_dir = tmp_path/'training'
    config = Config(output=str(run_dir), num_envs=1, total_steps=4, rollout_steps=4,
                    history_length=4, prediction_horizon=2, hidden_dim=8, latent_dim=2,
                    epochs=1, minibatch_size=4, aux_updates=0, eval_every=4, eval_episodes=1)
    run(config)
    original_manifest = json.loads((run_dir/'manifest.json').read_text())
    create = train_module.create_env
    rollout_environments = []

    def actual_rollout_env(config):
        env = create(config)
        env.metadata = {**env.metadata, 'actual_evaluation_marker': 'rollout-environment-only'}
        env.runtime_info = {'game_loop': 7, 'process_seed': 20001}
        rollout_environments.append(env)
        return env

    # evaluate.py has its own imported create_env for policy dimension setup;
    # train.evaluate must collect metadata from this distinct rollout factory.
    monkeypatch.setattr(train_module, 'create_env', actual_rollout_env)
    output = tmp_path/'evaluation.json'
    monkeypatch.setattr(sys, 'argv', ['dms.evaluate', '--checkpoint', str(run_dir/'latest.pt'),
                                     '--episodes', '1', '--output', str(output)])
    main()
    record = json.loads(output.read_text())
    assert len(rollout_environments) == 1
    assert record['manifest'] == original_manifest
    assert 'actual_evaluation_marker' not in record['manifest']['environment']
    assert record['evaluation_environment']['actual_evaluation_marker'] == 'rollout-environment-only'
    assert record['evaluation_runtime']['initial_environment']['game_loop'] == 7
    report = aggregate_evaluations([record])
    assert not report['provenance']['evaluation_environments_match_training']
    assert any('evaluation environment differs' in warning for warning in report['provenance']['warnings'])


def test_evaluation_metadata_is_an_independent_snapshot_and_env_closes_on_failure(monkeypatch):
    import dms.train as train_module

    config = Config(num_envs=1, total_steps=4, rollout_steps=4, history_length=4,
                    prediction_horizon=2, hidden_dim=8, latent_dim=2)
    env = train_module.create_env(config)
    learner = train_module.Learner(config, env)
    env.runtime_info = {'game_loop': 3}
    original_metadata = deepcopy(env.metadata)
    close_calls = []

    def close():
        close_calls.append(True)
        env.metadata['changed_on_close'] = True
        env.runtime_info['game_loop'] = -1

    def fail_step(actions):
        raise RuntimeError('manufactured evaluation step failure')

    monkeypatch.setattr(env, 'step', fail_step)
    monkeypatch.setattr(env, 'close', close)
    monkeypatch.setattr(train_module, 'create_env', lambda config: env)
    metadata = {}
    with pytest.raises(RuntimeError, match='manufactured'):
        train_module.evaluate(learner, 1, 20001, metadata=metadata)
    assert close_calls == [True]
    assert metadata['environment'] == original_metadata
    assert metadata['runtime']['initial_environment']['game_loop'] == 3
    assert metadata['runtime']['final_environment']['game_loop'] == 3
