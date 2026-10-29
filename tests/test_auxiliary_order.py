"""Auxiliary-first PPO must refresh local inputs without rewriting behavior."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

import dms.train as training
from dms.train import (Config, prepare_intention_inputs, refresh_intention_context,
                       update, update_intentions)
from test_sequential_update import assert_same, batch, model, ordered


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def auxiliary_model(*, discrete=True, mode='independent', fusion='concat', **settings):
    learner = model(algorithm='dms', discrete=discrete, mode=mode,
                    intention_fusion=fusion, history_length=4,
                    prediction_horizon=2, intention_update_order='before_ppo', **settings)
    learner.config.aux_updates = 2
    learner.config.aux_batch_size = 4
    learner.config.intention_lr = .03
    for optimizer in learner.aux_optimizers:
        optimizer.param_groups[0]['lr'] = .03
    if fusion == 'residual':
        for agent in learner.agents:
            with torch.no_grad():
                agent.actor_intention_projection.weight.normal_(std=.2)
                agent.critic_intention_projection.weight.normal_(std=.2)
    return learner


def auxiliary_batch(learner):
    data = batch(learner)
    h, n, a = learner.config.history_length, learner.num_agents, learner.action_dim
    rng = np.random.default_rng(35)
    data['history'] = rng.normal(size=(2, 2, h, n, n, a)).astype(np.float32)
    data['visible'] = rng.random((2, 2, h, n, n)) > .2
    data['future'] = rng.normal(size=(2, 2, n, n, a)).astype(np.float32)
    data['future_visible'] = np.ones((2, 2, n, n), bool)
    if learner.discrete:
        for key in ['history', 'future']:
            labels = rng.integers(a, size=data[key].shape[:-1])
            data[key] = np.eye(a, dtype=np.float32)[labels]
    data['ends'][:] = False
    for i in range(n):
        inputs = prepare_intention_inputs(learner, data, i, force=True)
        with torch.no_grad():
            data['context'][:, :, i] = learner.intentions[i].features(*inputs).reshape(2, 2, n, -1)
    with torch.no_grad():
        actions, log_prob, values = learner.action(
            data['obs'].flatten(0, 1), data['context'].flatten(0, 1),
            data['gates'].flatten(0, 1),
            data['available'].flatten(0, 1) if learner.discrete else None)
    for key, value in [('actions', actions), ('log_prob', log_prob), ('values', values)]:
        data[key] = value.reshape(2, 2, *value.shape[1:])
    return data


@pytest.mark.parametrize('value', [None, True, '', 'before', 'after'])
def test_schedule_is_explicit(value):
    assert Config().intention_update_order == 'after_ppo'
    with pytest.raises(ValueError, match='intention_update_order'):
        Config(intention_update_order=value).validate()


def test_joint_prior_and_auxiliary_first_require_separate_validation():
    with pytest.raises(ValueError, match='intention_rl_lr=0'):
        Config(intention_update_order='before_ppo', intention_rl_lr=1e-4).validate()
    Config(algorithm='ippo', intention_update_order='before_ppo', intention_rl_lr=1e-4).validate()


@pytest.mark.parametrize('discrete', [False, True])
@pytest.mark.parametrize('fusion', ['concat', 'residual'])
@pytest.mark.parametrize('mode', ['independent', 'sequential'])
def test_each_observer_fits_auxiliary_then_ppo_with_refreshed_detached_inputs(
        discrete, fusion, mode, monkeypatch):
    learner = auxiliary_model(discrete=discrete, fusion=fusion, mode=mode)
    data = auxiliary_batch(learner)
    original = deepcopy(data)
    original_auxiliary = training.update_intentions
    original_refresh = training.refresh_intention_context
    calls, expected_context = [], {}

    def auxiliary(learner, samples, *, agent_index=None):
        assert agent_index is not None
        calls.append(('aux', agent_index))
        return original_auxiliary(learner, samples, agent_index=agent_index)

    def refresh(learner, i, cached, gates, inputs):
        assert calls[-1] == ('aux', i)
        result = original_refresh(learner, i, cached, gates, inputs)
        assert not result.requires_grad
        with torch.no_grad():
            expected = learner.intentions[i].features(*inputs)
        assert_same(result, expected)
        assert not torch.equal(cached, result)
        expected_context[i] = result
        return result

    for i, agent in enumerate(learner.agents):
        original_distribution = agent.distribution

        def distribution(obs, context, gates, available=None, *, i=i, original=original_distribution):
            assert i in expected_context
            # Rows may be permuted, but every actual PPO context must be one
            # of the post-auxiliary means, with no autograd path to the CVAE.
            assert not context.requires_grad
            assert all(any(torch.equal(row, candidate) for candidate in expected_context[i])
                       for row in context)
            calls.append(('policy', i))
            return original(obs, context, gates, available)
        monkeypatch.setattr(agent, 'distribution', distribution)
    monkeypatch.setattr(training, 'update_intentions', auxiliary)
    monkeypatch.setattr(training, 'refresh_intention_context', refresh)
    result = update(learner, data)
    assert_same(original, data)
    assert result['forecast_target_count'] > 0
    assert sum(kind == 'aux' for kind, i in calls) == learner.num_agents
    for i in range(learner.num_agents):
        assert next(j for j, value in enumerate(calls) if value == ('aux', i)) < next(
            j for j, value in enumerate(calls) if value == ('policy', i))
        assert not any(p in learner.optimizers[i].state for p in learner.intentions[i].parameters())


@pytest.mark.parametrize('discrete', [False, True])
def test_sequential_later_observers_stay_at_behavior_and_factor_includes_encoder_change(discrete, monkeypatch):
    learner = auxiliary_model(discrete=discrete, mode='sequential', actor_lr=0, critic_lr=0)
    data = auxiliary_batch(learner)
    ordered(monkeypatch)
    before = deepcopy(learner.intentions.state_dict())
    original_auxiliary = training.update_intentions
    original_correction = training.advance_sequential_correction
    factors = []

    def auxiliary(learner, samples, *, agent_index):
        i = agent_index
        # Unprocessed priors and actor parameters still reproduce the behavior
        # likelihood, even after predecessor auxiliary/PPO updates.
        with torch.no_grad():
            inputs = prepare_intention_inputs(learner, samples, i, force=True)
            context = learner.intentions[i].features(*inputs)
            available = samples['available'][:, :, i].flatten(0, 1) if discrete else None
            dist = learner.agents[i].distribution(samples['obs'][:, :, i].flatten(0, 1),
                context, samples['gates'][:, :, i].flatten(0, 1), available)
            actions = samples['actions'][:, :, i].flatten(0, 1)
            if not discrete:
                actions = actions[:, :learner.action_dims[i]]
            assert_same(dist.log_prob(actions), samples['log_prob'][:, :, i].flatten())
        for k, value in learner.intentions[i].state_dict().items():
            assert_same(before[f'{i}.{k}'], value)
        return original_auxiliary(learner, samples, agent_index=i)

    def correction(learner, samples, i, log_correction, *, updated_context):
        available = samples['available'][:, :, i].flatten(0, 1) if discrete else None
        with torch.no_grad():
            dist = learner.agents[i].distribution(samples['obs'][:, :, i].flatten(0, 1),
                updated_context, samples['gates'][:, :, i].flatten(0, 1), available)
            actions = samples['actions'][:, :, i].flatten(0, 1)
            if not discrete:
                actions = actions[:, :learner.action_dims[i]]
            expected = dist.log_prob(actions) - samples['log_prob'][:, :, i].flatten()
        actual = original_correction(learner, samples, i, log_correction, updated_context=updated_context)
        assert_same(actual, log_correction + expected)
        assert expected.abs().max() > 1e-6  # Actor LR is zero: encoder shift alone.
        factors.append(actual)
        return actual

    monkeypatch.setattr(training, 'update_intentions', auxiliary)
    monkeypatch.setattr(training, 'advance_sequential_correction', correction)
    update(learner, data)
    assert len(factors) == learner.num_agents


def test_refresh_does_not_read_future_targets_or_change_filter_gates_or_unused_rows():
    learner = auxiliary_model(intention_threshold=0)
    data = auxiliary_batch(learner)
    data['gates'][0, :, 0] = False
    cached = data['context'][:, :, 0].flatten(0, 1)
    gates = data['gates'][:, :, 0].flatten(0, 1)
    inputs = prepare_intention_inputs(learner, data, 0, force=True)
    original_inputs, original_gates = deepcopy(inputs), gates.clone()
    with torch.no_grad():
        learner.intentions[0].gaussian_head.bias[:learner.latent_dim].add_(2)
    expected = refresh_intention_context(learner, 0, cached, gates, inputs)
    assert_same(inputs, original_inputs)
    data['future'][:] = np.nan
    data['history'][~data['visible']] = np.nan
    actual = refresh_intention_context(learner, 0, cached, gates, inputs)
    assert_same(expected, actual)
    # The tensor view may share the caller's history storage; poisoning that
    # storage above is intentional. Refresh itself must not change visibility.
    assert_same(inputs[1], original_inputs[1])
    assert_same(gates, original_gates)
    assert_same(actual[:2], cached[:2])
    assert not torch.equal(actual[2:], cached[2:])


@pytest.mark.parametrize('reason', ['disabled', 'no_windows', 'ippo'])
@pytest.mark.parametrize('mode', ['independent', 'sequential'])
def test_inapplicable_schedule_replays_default_state_metrics_and_rng(reason, mode):
    learner = auxiliary_model(mode=mode)
    data = auxiliary_batch(learner)
    if reason == 'disabled':
        learner.config.aux_updates = 0
    elif reason == 'no_windows':
        data['ends'][:] = True
    else:
        learner = model(mode=mode, intention_update_order='before_ppo')
        data = batch(learner)
    other = deepcopy(learner)
    other.config.intention_update_order = 'after_ppo'
    results = []
    for current in [learner, other]:
        torch.manual_seed(91); np.random.seed(91); random.seed(91)
        metrics = update(current, data)
        results.append((metrics, torch.get_rng_state().clone(), np.random.get_state(), random.getstate()))
    assert_same(results[0], results[1])
    assert_same(learner.agents.state_dict(), other.agents.state_dict())
    assert_same(learner.intentions.state_dict(), other.intentions.state_dict())
    assert_same([o.state_dict() for o in learner.optimizers], [o.state_dict() for o in other.optimizers])
    assert_same([o.state_dict() for o in learner.aux_optimizers], [o.state_dict() for o in other.aux_optimizers])
