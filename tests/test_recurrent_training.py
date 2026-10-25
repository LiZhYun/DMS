"""Bounded recurrent PPO integration; only toy/in-memory environments are used."""
import copy
import json
import sys

import numpy as np
import pytest
import torch

import dms.train as training
from dms.envs import EnvStep
from dms.observation_history import ObservationHistoryEnv
from dms.policy import RunningMoments


@pytest.fixture
def setup_rollout():
    opened = []

    def build(**overrides):
        options = dict(algorithm='ippo', num_envs=2, rollout_steps=6, total_steps=12,
                       epochs=1, minibatch_size=64, hidden_dim=8, latent_dim=2,
                       history_length=4, prediction_horizon=2, policy_history_length=3,
                       aux_updates=1, aux_batch_size=3, eval_episodes=2, seed=29)
        options.update(overrides)
        config = training.Config(**options)
        config.validate()
        training.seed_everything(config.seed)
        envs = [training.create_env(config) for _ in range(config.num_envs)]
        opened.extend(envs)
        learner = training.Learner(config, envs[0])
        states = [env.reset(seed=31+i) for i, env in enumerate(envs)]
        history, visible = training.empty_history(learner, len(envs))
        return learner, envs, states, history, visible

    yield build
    for env in opened:
        env.close()


def collect(learner, envs, states, history, visible, **kwargs):
    return training.collect(learner, envs, states, history, visible,
                            kwargs.pop('episodes', 0),
                            kwargs.pop('episode_returns', np.zeros(len(envs))), **kwargs)


@pytest.mark.parametrize('algorithm', ['ippo', 'dms'])
def test_collected_windows_replay_exact_ppo_likelihoods_and_train_both_encoders(setup_rollout, algorithm):
    learner, envs, states, history, visible = setup_rollout(algorithm=algorithm)
    data, *_ = collect(learner, envs, states, history, visible)
    assert data['obs'].shape[-2:] == (3, envs[0].obs_dim+1)
    cached = data['obs'].clone()
    for observer, (agent, optimizer) in enumerate(zip(learner.agents, learner.optimizers)):
        dist = agent.distribution(data['obs'][:, :, observer].flatten(0, 1),
                                  data['context'][:, :, observer].flatten(0, 1),
                                  data['gates'][:, :, observer].flatten(0, 1),
                                  data['available'][:, :, observer].flatten(0, 1))
        replay = dist.log_prob(data['actions'][:, :, observer].flatten())
        expected = data['log_prob'][:, :, observer].flatten()
        torch.testing.assert_close(replay, expected, atol=2e-6, rtol=1e-6)
        critic = {id(p) for p in agent.critic_parameters()}
        actor_group, critic_group = optimizer.param_groups
        assert {id(p) for p in critic_group['params']} == critic
        assert {id(p) for p in agent.actor_encoder.parameters()} <= {id(p) for p in actor_group['params']}
        assert not critic & {id(p) for p in actor_group['params']}
        assert critic_group['lr'] == learner.config.critic_lr
        assert actor_group['lr'] == learner.config.actor_lr
    metrics = training.update(learner, data)
    assert np.isfinite(list(metrics.values())).all()
    # One full minibatch per agent: these are the pre-update PPO ratios, not a
    # second rollout or regenerated intention contexts under updated parameters.
    assert abs(metrics['approx_kl']) < 1e-6
    torch.testing.assert_close(data['obs'], cached, atol=0, rtol=0)
    for agent, optimizer in zip(learner.agents, learner.optimizers):
        for encoder in (agent.actor_encoder, agent.critic_encoder):
            assert all(parameter in optimizer.state for parameter in encoder.parameters())


class ScriptedLocalEnvironment:
    """Two local streams with explicit timeout and optional no-op-only death."""
    n_agents, obs_dim, action_dim, action_dims, discrete = 2, 3, 3, [3, 3], True
    action_visibility = np.ones((2, 2, 3), dtype=bool)
    metadata = {'benchmark_track': 'in_memory_local_test'}

    def __init__(self, death=False):
        self.death, self.time, self.episodes = death, 0, 0
        self.previous = None

    def snapshot(self):
        alive = np.array([not self.death or self.time < 2, True])
        mask = np.array([[False, True, True]]*2)
        mask[~alive] = [True, False, False]
        raw = np.array([[self.episodes+self.time/10, 1, 0],
                        [self.episodes+self.time/10, 0, 1]], np.float32)
        raw[~alive] = 0
        visible = alive[:, None] & alive[None]
        observed = np.zeros((2, 2, 3), np.float32)
        if self.previous is not None:
            observed = np.where(visible[..., None], np.eye(3, dtype=np.float32)[self.previous][None], 0)
        else:
            visible[:] = False
        return EnvStep(raw, np.ones(2, np.float32), np.zeros(2, bool),
                       np.full(2, self.time == 4), mask, observed, visible,
                       {'battle_won': False, 'time': self.time})

    def reset(self, seed=None):
        self.episodes += 1
        self.time, self.previous = 0, None
        return self.snapshot()

    def step(self, actions):
        assert self.time < 4
        assert self.snapshot().action_mask[np.arange(2), actions].all()
        self.previous = actions.copy()
        self.time += 1
        return self.snapshot()

    def close(self):
        pass


def test_timeout_bootstrap_uses_final_window_and_next_decision_uses_reset_window(setup_rollout, monkeypatch):
    monkeypatch.setattr(training, 'make_env', lambda *args, **kwargs: ScriptedLocalEnvironment())
    learner, envs, states, history, visible = setup_rollout(num_envs=1, rollout_steps=5)
    calls = []
    original = learner.contexts

    def record(states, *args, **kwargs):
        result = original(states, *args, **kwargs)
        calls.append((states[0].obs.copy(), result))
        return result

    monkeypatch.setattr(learner, 'contexts', record)
    data, *_ = collect(learner, envs, states, history, visible)
    final_raw, final_inputs = next(row for row in calls if row[0][0, -1, 0] == np.float32(1.4))
    np.testing.assert_allclose(final_raw[0, :, 0], [1.2, 1.3, 1.4])
    assert final_raw[:, :, -1].all()
    expected = torch.stack([agent.value(final_inputs[0][:, i], final_inputs[1][:, i],
                                        final_inputs[2][:, i])
                            for i, agent in enumerate(learner.agents)], 1)
    torch.testing.assert_close(data['next_values'][3], expected, atol=0, rtol=0)
    assert data['truncated'][3].all() and not data['terminated'][3].any()
    np.testing.assert_array_equal(data['raw_obs'][4, 0, 0, :, 0], [0, 0, 2])
    np.testing.assert_array_equal(data['raw_obs'][4, 0, 0, :, -1], [0, 0, 1])
    np.testing.assert_allclose(final_raw[0, :, 0], [1.2, 1.3, 1.4])


def test_raw_windows_survive_updates_and_renormalize_without_mutating_cached_rollout(setup_rollout):
    learner, envs, states, history, visible = setup_rollout(num_envs=1, rollout_steps=3,
                                                          policy_history_length=4)
    data, states, history, visible, episodes, returns, _ = collect(learner, envs, states, history, visible)
    old_raw, old_normalized = data['raw_obs'].copy(), data['obs'].clone()
    current_raw = states[0].obs.copy()
    before = learner.contexts(states, history, visible)[0]
    training.update(learner, data)
    expected = learner.contexts(states, history, visible)[0]
    assert not torch.allclose(before[..., :-1], expected[..., :-1])
    for observer, agent in enumerate(learner.agents):
        moments = RunningMoments(envs[0].obs_dim)
        moments.update(learner.tensor(old_raw[:, :, observer, -1, :-1]))
        for key in moments.state_dict():
            torch.testing.assert_close(agent.obs_moments.state_dict()[key], moments.state_dict()[key], atol=0, rtol=0)
    next_data, *_ = collect(learner, envs, states, history, visible, episodes=episodes,
                            episode_returns=returns, rollout_steps=1)
    torch.testing.assert_close(next_data['obs'][0], expected, atol=0, rtol=0)
    np.testing.assert_array_equal(next_data['raw_obs'][0, 0], current_raw)
    np.testing.assert_array_equal(states[0].obs, current_raw)
    np.testing.assert_array_equal(data['raw_obs'], old_raw)
    torch.testing.assert_close(data['obs'], old_normalized, atol=0, rtol=0)


def test_mock_smac_death_resets_own_window_and_survives_active_masked_ppo(setup_rollout, monkeypatch):
    import dms.smac_env as smac

    monkeypatch.setattr(smac, 'SMACEnv', lambda *args, **kwargs: ScriptedLocalEnvironment(death=True))
    learner, envs, states, history, visible = setup_rollout(env='smac', num_envs=1,
                                                          mask_inactive_agents=True)
    assert isinstance(envs[0], ObservationHistoryEnv) and envs[0].clear_dead_agents
    data, *_ = collect(learner, envs, states, history, visible)
    assert not data['raw_obs'][2:4, 0, 0].any()
    assert data['raw_obs'][2:4, 0, 1, :, -1].all()
    assert data['actions'][2:4, 0, 0].eq(0).all()
    np.testing.assert_array_equal(data['raw_obs'][4, 0, 0, :, -1], [0, 0, 1])
    metrics = training.update(learner, data)
    assert np.isfinite(list(metrics.values())).all()
    for observer, agent in enumerate(learner.agents):
        alive = data['available'][:, :, observer, 1:].any(-1)
        assert agent.obs_moments.count.item() == pytest.approx(alive.sum().item()+1e-4)


@pytest.mark.parametrize('length', [0, 3])
def test_checkpoint_cli_reconstructs_correct_evaluator_and_keeps_model_frozen(setup_rollout, tmp_path, monkeypatch, length):
    import dms.evaluate as evaluator

    learner, envs, states, history, visible = setup_rollout(num_envs=1, policy_history_length=length)
    data, *_ = collect(learner, envs, states, history, visible)
    training.update(learner, data)
    checkpoint_path, output = tmp_path/'checkpoint.pt', tmp_path/'evaluation.json'
    learner.save(checkpoint_path, 6, 0)
    if length == 0:
        # Simulate an archived feed-forward checkpoint predating the option.
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint['config'].pop('policy_history_length')
        torch.save(checkpoint, checkpoint_path)
    original_evaluate = evaluator.evaluate
    frozen_checks = []

    def checked_evaluate(restored, *args, **kwargs):
        before = copy.deepcopy(restored.agents.state_dict())
        records = original_evaluate(restored, *args, **kwargs)
        for key, value in restored.agents.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        frozen_checks.append(True)
        return records

    monkeypatch.setattr(evaluator, 'evaluate', checked_evaluate)
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--checkpoint', str(checkpoint_path),
                                     '--episodes', '2', '--seed', '701', '--output', str(output)])
    evaluator.main()
    report = json.loads(output.read_text())
    assert report['config']['policy_history_length'] == length
    assert [row['seed'] for row in report['results']] == [701, 702]
    assert all(row['length'] == 32 for row in report['results'])
    assert frozen_checks == [True]
    if length:
        assert report['evaluation_environment']['policy_observation_history']['length'] == length
    else:
        assert 'policy_observation_history' not in report['evaluation_environment']
