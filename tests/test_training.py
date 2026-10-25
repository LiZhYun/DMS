import json
import pytest
import torch

from dms.train import Config, Learner, run


def test_unknown_environment_cannot_silently_run_the_toy_task():
    from dms.train import create_env
    config = Config(env='mamujco')
    with pytest.raises(ValueError, match='env must'):
        config.validate()
    with pytest.raises(ValueError, match='unknown environment'):
        create_env(config)


@pytest.mark.parametrize('algorithm,decoder_history', [('ippo',True),('dms',True),('dms',False)])
def test_tiny_rollout_update_evaluation_and_checkpoint(tmp_path, algorithm, decoder_history):
    output = tmp_path / algorithm
    config = Config(algorithm=algorithm, decoder_history=decoder_history, output=str(output), num_envs=2,
                    rollout_steps=8, total_steps=20, epochs=1, minibatch_size=16,
                    hidden_dim=16, latent_dim=2, history_length=4,
                    prediction_horizon=2, aux_updates=1, aux_batch_size=4,
                    eval_every=20, eval_episodes=3, seed=7)
    run(config)
    metrics = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    evaluation = [json.loads(line) for line in (output/'evaluation.jsonl').read_text().splitlines()]
    assert [row['steps'] for row in metrics] == [16,20]
    assert len(evaluation) == 3  # one worker must complete more than one episode
    assert [row['seed'] for row in evaluation] == [10001, 10002, 10003]
    checkpoint = torch.load(output/'latest.pt', weights_only=False)
    assert checkpoint['steps'] == 20
    assert all(optimizer['state'] for optimizer in checkpoint['optimizers'])
    if algorithm == 'dms':
        assert 'intention_loss' in metrics[-1]
        assert 'forecast_nll' in metrics[-1]
        assert all(optimizer['state'] for optimizer in checkpoint['aux_optimizers'])
    assert not checkpoint['environment_state_saved']
    with pytest.raises(FileExistsError):
        run(config)


def test_heterogeneous_agents_have_no_padded_policy_coordinates():
    import numpy as np
    from types import SimpleNamespace
    env = SimpleNamespace(n_agents=2,action_dim=3,action_dims=[1,3],obs_dim=4,
                          discrete=False,action_visibility=np.ones((2,2,3),bool))
    learner = Learner(Config(algorithm='ippo'),env)
    obs = torch.zeros(5,2,4)
    contexts = torch.zeros(5,2,2,0)
    gates = torch.zeros(5,2,2,dtype=torch.bool)
    actions, log_prob, _ = learner.action(obs,contexts,gates,None)
    assert actions.shape == (5,2,3)
    assert actions[:,0,1:].eq(0).all()
    assert learner.agents[0].log_std.shape == (1,)
    distribution = learner.agents[0].distribution(obs[:,0],contexts[:,0],gates[:,0])
    torch.testing.assert_close(log_prob[:,0],distribution.log_prob(actions[:,0,:1]))


def test_real_mujoco_continuous_update_with_partial_sensors():
    import numpy as np
    from dms.train import create_env, empty_history, collect, update
    config = Config(env='mamujoco',scenario='HalfCheetah',agent_conf='3x2',agent_obsk=1,
                    num_envs=1,rollout_steps=4,total_steps=4,epochs=1,minibatch_size=4,
                    hidden_dim=8,latent_dim=2,history_length=4,prediction_horizon=2,
                    aux_updates=1,aux_batch_size=2)
    env = create_env(config)
    try:
        states = [env.reset(seed=3)]
        learner = Learner(config,env)
        history, visible = empty_history(learner,1)
        data, *_ = collect(learner,[env],states,history,visible,0,np.zeros(1))
        metrics = update(learner,data)
        assert np.isfinite(list(metrics.values())).all()
        assert 'intention_loss' in metrics
        assert 'forecast_rmse' in metrics
        assert 'persistence_rmse' in metrics
        assert data['actions'].shape == (4,1,3,2)
        assert data['gates'].any()  # observable peers enter the method
    finally:
        env.close()


def test_rollout_reuses_exact_successor_inputs_until_reset(monkeypatch):
    import numpy as np
    from dms.train import create_env, empty_history, collect
    config = Config(num_envs=1,rollout_steps=40,total_steps=40,
                    hidden_dim=8,latent_dim=2,history_length=4,prediction_horizon=2,
                    warmup_episodes=1)
    env = create_env(config)
    try:
        learner = Learner(config,env)
        states = [env.reset(seed=3)]
        history, visible = empty_history(learner,1)
        calls = []
        original = learner.contexts
        def record_contexts(*args, **kwargs):
            inputs = original(*args, **kwargs)
            calls.append(inputs)
            return inputs
        monkeypatch.setattr(learner,'contexts',record_contexts)
        data,*_ = collect(learner,[env],states,history,visible,0,np.zeros(1))
        # One initial call, one successor call/step, plus one after the toy's
        # step32 reset. That reset also crosses the intention warmup boundary.
        assert len(calls) == 42
        torch.testing.assert_close(data['obs'][1],calls[1][0])
        torch.testing.assert_close(data['context'][1],calls[1][1])
        assert not data['gates'][:32].any()
        assert data['gates'][33:].any()
    finally:
        env.close()


def test_parallel_steps_preserve_complete_training_and_reset_trajectory(tmp_path):
    results = []
    for parallel in (False, True):
        output = tmp_path / str(parallel)
        config = Config(output=str(output), num_envs=2, parallel_env_steps=parallel,
                        rollout_steps=20, total_steps=80, epochs=1, minibatch_size=40,
                        hidden_dim=8, latent_dim=2, history_length=4, prediction_horizon=2,
                        aux_updates=1, aux_batch_size=4, warmup_episodes=1,
                        eval_every=80, eval_episodes=2, seed=7)
        run(config)
        metrics = [json.loads(row) for row in (output/'metrics.jsonl').read_text().splitlines()]
        for row in metrics:
            row.pop('elapsed_seconds')
        results.append((metrics, (output/'evaluation.jsonl').read_text(),
                        torch.load(output/'latest.pt', weights_only=False)))
    assert results[0][:2] == results[1][:2]
    assert results[0][2]['episodes'] >= 2  # Includes resets and a warmup-phase change.
    for module in ('agents', 'intentions'):
        for name, value in results[0][2][module].items():
            torch.testing.assert_close(value, results[1][2][module][name], rtol=0, atol=0)


def test_failed_run_records_last_completed_update_and_closes_environments(tmp_path, monkeypatch):
    import dms.train as training
    original_collect, original_create = training.collect, training.create_env
    calls, closed = [], []
    def fail_second_collection(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('simulated simulator failure')
        return original_collect(*args, **kwargs)
    def create(config):
        env = original_create(config)
        original_close = env.close
        def close():
            closed.append(env)
            original_close()
        env.close = close
        return env
    monkeypatch.setattr(training, 'collect', fail_second_collection)
    monkeypatch.setattr(training, 'create_env', create)
    config = Config(algorithm='ippo', output=str(tmp_path/'failed'), num_envs=2,
                    parallel_env_steps=True, rollout_steps=4, total_steps=16,
                    epochs=1, hidden_dim=8, eval_every=100)
    with pytest.raises(RuntimeError, match='simulated simulator failure'):
        run(config)
    manifest = json.loads((tmp_path/'failed'/'manifest.json').read_text())
    assert manifest['status'].startswith('failed')
    assert manifest['completed_steps'] == 8
    assert manifest['failure'] == {'type':'RuntimeError', 'message':'simulated simulator failure'}
    assert len(closed) == 2
