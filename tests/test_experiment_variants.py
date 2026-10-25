"""Configuration plumbing for explicit variants; no native simulator is loaded."""
from dataclasses import asdict
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import dms.train as training


def local_environment(discrete=False):
    return SimpleNamespace(n_agents=2, obs_dim=4, action_dim=3,
                           action_dims=[3, 3] if discrete else [1, 3], discrete=discrete,
                           action_visibility=np.ones((2, 2, 3), dtype=bool))


def test_missing_options_restore_original_defaults_and_policy_checkpoint_layout():
    current = training.Config(algorithm='ippo', hidden_dim=8)
    old_config = asdict(current)
    old_config.pop('initial_log_std')
    old_config.pop('ant_contact_forces')
    restored = training.Config(**old_config)
    restored.validate()
    assert restored.initial_log_std == -.5 and restored.ant_contact_forces == 'native'
    torch.manual_seed(51)
    expected = training.Learner(current, local_environment())
    torch.manual_seed(51)
    actual = training.Learner(restored, local_environment())
    assert expected.agents.state_dict().keys() == actual.agents.state_dict().keys()
    for key, value in expected.agents.state_dict().items():
        torch.testing.assert_close(actual.agents.state_dict()[key], value, atol=0, rtol=0)
    actual.agents.load_state_dict(expected.agents.state_dict())


def test_config_scale_reaches_each_real_action_dimension_and_checkpoint(tmp_path):
    config = training.Config(algorithm='ippo', hidden_dim=8, initial_log_std=-1,
                             env='mamujoco', mamujoco_backend='legacy', scenario='Ant',
                             agent_conf='4x2', ant_contact_forces='computed')
    config.validate()
    learner = training.Learner(config, local_environment())
    obs = torch.zeros(5, 2, 4)
    context, gates = torch.zeros(5, 2, 2, 0), torch.zeros(5, 2, 2, dtype=torch.bool)
    actions, log_prob, _ = learner.action(obs, context, gates, None)
    assert actions.shape == (5, 2, 3) and actions[:, 0, 1:].eq(0).all()
    for observer, width in enumerate((1, 3)):
        agent = learner.agents[observer]
        assert agent.log_std.shape == (width,)
        distribution = agent.distribution(obs[:, observer], context[:, observer], gates[:, observer])
        torch.testing.assert_close(distribution.stddev, torch.full((5, width), math.exp(-1)))
        torch.testing.assert_close(log_prob[:, observer], distribution.log_prob(actions[:, observer, :width]))
    path = tmp_path/'variants.pt'
    learner.save(path, steps=8, episodes=0)
    checkpoint = torch.load(path, weights_only=False)
    saved = training.Config(**checkpoint['config'])
    saved.validate()
    assert saved.initial_log_std == -1 and saved.ant_contact_forces == 'computed'
    assert json.loads(json.dumps(asdict(saved))) == checkpoint['config']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'),
                                  -5.001, 2.001, True, False, '-1', None])
def test_config_rejects_invalid_initial_gaussian_scale(value):
    with pytest.raises(ValueError, match='initial_log_std'):
        training.Config(initial_log_std=value).validate()


@pytest.mark.parametrize('value', [-5, -1, 2])
def test_config_accepts_finite_initial_gaussian_scale_bounds(value):
    training.Config(initial_log_std=value).validate()


@pytest.mark.parametrize('value', [None, True, 0, 'auto', 'Computed'])
def test_config_rejects_unknown_contact_force_modes(value):
    with pytest.raises(ValueError, match='ant_contact_forces'):
        training.Config(ant_contact_forces=value).validate()


@pytest.mark.parametrize('env,backend,scenario', [
    ('toy', 'legacy', 'Ant'),
    ('smac', 'legacy', 'Ant'),
    ('mamujoco', 'modern', 'Ant'),
    ('mamujoco', 'legacy', 'HalfCheetah'),
    ('mamujoco', 'legacy', 'Walker2d'),
    ('mamujoco', 'legacy', 'ManyAgentSwimmer'),
])
def test_computed_contact_forces_require_historical_ant(env, backend, scenario):
    config = training.Config(env=env, mamujoco_backend=backend, scenario=scenario,
                             ant_contact_forces='computed')
    with pytest.raises(ValueError, match='historical legacy Ant'):
        config.validate()


@pytest.mark.parametrize('mode', ['native', 'computed'])
def test_legacy_factory_receives_selected_contact_mode_before_optional_history_wrapper(monkeypatch, mode):
    import dms.legacy_env as legacy
    from dms.observation_history import ObservationHistoryEnv

    calls, base = [], local_environment()

    def construct(*args, **kwargs):
        calls.append((args, kwargs))
        return base

    monkeypatch.setattr(legacy, 'LegacyMaMuJoCoEnv', construct)
    config = training.Config(env='mamujoco', mamujoco_backend='legacy', scenario='Ant',
                             agent_conf='4x2', ant_contact_forces=mode, policy_history_length=3)
    config.validate()
    env = training.create_env(config)
    assert isinstance(env, ObservationHistoryEnv) and env.env is base
    assert calls == [(('Ant', '4x2', 1), {'ant_contact_forces': mode})]


def test_default_contact_mode_does_not_change_modern_factory_arguments(monkeypatch):
    calls, base = [], local_environment()

    def make(*args, **kwargs):
        calls.append((args, kwargs))
        return base

    monkeypatch.setattr(training, 'make_env', make)
    config = training.Config(env='mamujoco', scenario='HalfCheetah', agent_conf='2x3')
    config.validate()
    assert training.create_env(config) is base
    assert calls == [(('mamujoco',), {'scenario': 'HalfCheetah', 'agent_conf': '2x3', 'agent_obsk': 1})]


def test_discrete_learner_remains_identical_when_valid_gaussian_option_changes():
    learners = []
    for value in (-.5, -1):
        config = training.Config(algorithm='ippo', hidden_dim=8, initial_log_std=value)
        config.validate()
        torch.manual_seed(52)
        learners.append(training.Learner(config, local_environment(discrete=True)))
    original, changed = learners
    assert all(agent.log_std is None for agent in changed.agents)
    assert changed.agents.state_dict().keys() == original.agents.state_dict().keys()
    for key, value in original.agents.state_dict().items():
        torch.testing.assert_close(changed.agents.state_dict()[key], value, atol=0, rtol=0)
