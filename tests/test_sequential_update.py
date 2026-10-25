"""Sequential PPO likelihood correction and independent-mode regression checks."""
from copy import deepcopy
from types import SimpleNamespace
import json
import random

import numpy as np
import pytest
import torch

import dms.train as training
from dms.train import (Config, Learner, advance_sequential_correction,
                       sequential_factor, update)


def model(*, mode='independent', discrete=True, env='toy', mask=False,
          action_dims=(3, 3), history=0, algorithm='ippo', **settings):
    config = Config(algorithm=algorithm, env=env, update_mode=mode,
                    mask_inactive_agents=mask, hidden_dim=8, latent_dim=2,
                    epochs=1, minibatch_size=4, aux_updates=0, target_kl=0,
                    policy_history_length=history, **settings)
    config.validate()
    environment = SimpleNamespace(n_agents=2, action_dim=max(action_dims),
        action_dims=list(action_dims), obs_dim=4, discrete=discrete,
        action_visibility=np.ones((2, 2, max(action_dims)), bool))
    torch.manual_seed(17)
    return Learner(config, environment)


def batch(learner, alive=None):
    # Four flattened rows, laid out as [time=2, environment=2].
    c, agents = learner.config, learner.num_agents
    raw = torch.arange(4*agents*4, dtype=torch.float32).reshape(4, agents, 4)/31
    if c.policy_history_length:
        raw = raw[:, :, None].repeat(1, 1, c.policy_history_length, 1)
        raw = torch.cat((raw, torch.ones(*raw.shape[:-1], 1)), -1)
        raw[:, :, 0, -1] = 0
    obs = torch.stack([agent.normalize_obs(raw[:, i])
                       for i, agent in enumerate(learner.agents)], 1)
    context = torch.linspace(-.5, .5, max(1, 4*agents*agents*learner.latent_dim))
    context = (context.reshape(4, agents, agents, learner.latent_dim) if learner.latent_dim
               else torch.zeros(4, agents, agents, 0))
    gates = (~torch.eye(agents, dtype=torch.bool))[None].repeat(4, 1, 1)
    available = None
    if learner.discrete:
        active = torch.ones(4, agents, dtype=torch.bool) if alive is None else torch.as_tensor(alive)
        available = torch.zeros(4, agents, learner.action_dim, dtype=torch.bool)
        available[..., 0] = ~active
        available[..., 1:] = active[..., None]
    with torch.no_grad():
        actions, log_prob, values = learner.action(obs, context, gates, available)
    def time_env(value):
        return value.reshape(2, 2, *value.shape[1:]).detach()
    data = dict(obs=time_env(obs), raw_obs=time_env(raw).numpy(),
                context=time_env(context), gates=time_env(gates), actions=time_env(actions),
                log_prob=time_env(log_prob), values=time_env(values),
                next_values=torch.zeros(2, 2, agents),
                rewards=np.arange(8, dtype=np.float32).reshape(2, 2, agents)/3,
                terminated=np.ones((2, 2, agents), bool),
                truncated=np.zeros((2, 2, agents), bool), ends=np.ones((2, 2), bool))
    if available is not None:
        data['available'] = time_env(available)
    return data


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def ordered(monkeypatch):
    # Agent order [0,1] and deterministic minibatch order, with no RNG draws.
    monkeypatch.setattr(torch, 'randperm', lambda n, device=None: torch.arange(n, device=device))


def fixed_advantages(monkeypatch):
    advantage = torch.tensor([[1., 10.], [2., 20.], [3., 30.], [4., 40.]]).reshape(2, 2, 2)
    monkeypatch.setattr(training, 'generalized_advantage_estimate',
        lambda *args, **kwargs: (advantage.clone(), torch.ones_like(advantage)))
    return advantage


def test_mode_is_explicit_and_validated():
    assert Config().update_mode == 'independent'
    with pytest.raises(ValueError, match='update_mode'):
        Config(update_mode='sequentail').validate()


@pytest.mark.parametrize('discrete,history', [(True, 0), (False, 0), (True, 3)])
def test_default_and_explicit_independent_weights_optimizers_metrics_and_rng_match(discrete, history):
    a = model(discrete=discrete, history=history)
    b = deepcopy(a)
    # Removing the serialized field models an old checkpoint: Config supplies
    # independent mode without adding a random agent-order draw.
    old_config = vars(b.config).copy()
    old_config.pop('update_mode')
    b.config = Config(**old_config)
    data = batch(a)
    results = []
    for learner in [a, b]:
        torch.manual_seed(41); np.random.seed(41); random.seed(41)
        metrics = update(learner, data)
        results.append((metrics, torch.get_rng_state().clone(), np.random.get_state(), random.getstate()))
    assert_same(results[0], results[1])
    assert_same(a.agents.state_dict(), b.agents.state_dict())
    assert_same([o.state_dict() for o in a.optimizers], [o.state_dict() for o in b.optimizers])
    assert 'sequential_agent_order' not in results[0][0]


def test_factor_multiplies_standardized_actor_surrogate_without_restandardizing(monkeypatch):
    learner = model(mode='sequential', actor_lr=0, critic_lr=0)
    data = batch(learner)
    advantage = fixed_advantages(monkeypatch)
    ordered(monkeypatch)
    log_factor = torch.tensor([1., 2., 3., 4.]).log().requires_grad_()
    monkeypatch.setattr(training, 'advance_sequential_correction',
                        lambda learner, data, i, correction: log_factor)
    metrics = update(learner, data)
    local = advantage[:, :, 1].flatten()
    standardized = (local-local.mean())/local.std(unbiased=False)
    expected = -(log_factor.detach().exp()*standardized).mean()/2
    assert metrics['policy_loss'] == pytest.approx(expected.item(), abs=2e-7)
    assert metrics['policy_loss'] != pytest.approx(0, abs=1e-4)
    assert log_factor.grad is None  # No derivative through predecessor policies.
    assert metrics['sequential_agent_order'] == [0, 1]
    assert metrics['sequential_factor_min'] == 1
    assert metrics['sequential_factor_max'] == 4
    assert metrics['sequential_factor_ess'] == pytest.approx(100/30)


def test_factor_never_weights_entropy_critic_or_value_moments(monkeypatch):
    independent = model(actor_lr=0, critic_lr=0, entropy_coef=.2, max_grad_norm=1e6)
    sequential = deepcopy(independent)
    sequential.config.update_mode = 'sequential'
    data = batch(independent)
    ordered(monkeypatch)
    target = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    monkeypatch.setattr(training, 'generalized_advantage_estimate',
                        lambda *args, **kwargs: (torch.zeros_like(target), target.clone()))
    monkeypatch.setattr(training, 'advance_sequential_correction',
                        lambda *args: torch.tensor([.25, 2., 7., 11.]).log())
    before, after = update(independent, data), update(sequential, data)
    for key in ['policy_loss', 'value_loss', 'entropy', 'approx_kl']:
        assert before[key] == after[key]
    for a, b in zip(independent.agents, sequential.agents):
        assert_same(a.state_dict(), b.state_dict())
        for left, right in zip(a.parameters(), b.parameters()):
            assert_same(left.grad, right.grad)


def test_joint_continuous_ratios_sum_action_coordinates_once_with_heterogeneous_dimensions():
    learner = model(mode='sequential', discrete=False, action_dims=(1, 3))
    data = batch(learner)
    data['actions'][:, :, 0, 1:] = 999  # Padding cannot enter the one-dimensional policy.
    expected = torch.zeros(4)
    correction = torch.zeros(4)
    for i, agent in enumerate(learner.agents):
        with torch.no_grad():
            agent.actor[-1].bias.add_(torch.linspace(.1, .3, learner.action_dims[i]))
            agent.log_std.add_(.1)
            obs = data['obs'][:, :, i].flatten(0, 1)
            contexts = data['context'][:, :, i].flatten(0, 1)
            gates = data['gates'][:, :, i].flatten(0, 1)
            dist = agent.distribution(obs, contexts, gates)
            actions = data['actions'][:, :, i].flatten(0, 1)[:, :learner.action_dims[i]]
            # Independent(Normal) already sums once across this agent's actual
            # action dimensions; use base Normal components as the oracle.
            joint = dist.base_dist.log_prob(actions).sum(-1)
            expected += joint-data['log_prob'][:, :, i].flatten()
        correction = advance_sequential_correction(learner, data, i, correction)
    assert correction.shape == (4,) and not correction.requires_grad
    torch.testing.assert_close(correction, expected, rtol=0, atol=0)
    torch.testing.assert_close(sequential_factor(correction), expected.exp())


@pytest.mark.parametrize('mask', [False, True])
def test_dead_predecessors_contribute_one_on_full_time_environment_grid(mask):
    learner = model(mode='sequential', env='smac', mask=mask)
    data = batch(learner, [[True, False], [False, True], [True, True], [False, False]])
    intended = [torch.tensor([2., 3., 5., 7.]), torch.tensor([11., 13., 17., 19.])]
    correction = torch.zeros(4)
    for i in range(2):
        data['log_prob'][:, :, i] -= intended[i].log().reshape(2, 2)
        correction = advance_sequential_correction(learner, data, i, correction)
    torch.testing.assert_close(correction.exp(), torch.tensor([2., 13., 85., 1.]))


def test_active_minibatches_index_predecessor_weights_by_original_rows(monkeypatch):
    learner = model(mode='sequential', env='smac', mask=True, actor_lr=0, critic_lr=0)
    data = batch(learner, [[True, False], [False, True], [True, True], [False, False]])
    fixed_advantages(monkeypatch)
    ordered(monkeypatch)
    monkeypatch.setattr(training, 'advance_sequential_correction',
                        lambda *args: torch.tensor([2., 1., 5., 1.]).log())
    result = update(learner, data)
    # Agent1's active original rows1,2 use weights1,5 and local advantages-1,+1.
    # Its actor loss is-2; agent0 has unit weights and zero mean actor loss.
    assert result['policy_loss'] == pytest.approx(-1., abs=2e-7)
    assert result['alive_fraction'] == .5


def test_all_dead_batch_skips_adam_and_keeps_identity_correction_after_momentum_exists():
    learner = model(mode='sequential', env='smac', mask=True)
    update(learner, batch(learner))
    agents = deepcopy(learner.agents.state_dict())
    optimizers = deepcopy([o.state_dict() for o in learner.optimizers])
    result = update(learner, batch(learner, [[False, False]]*4))
    assert_same(agents, learner.agents.state_dict())
    assert_same(optimizers, [o.state_dict() for o in learner.optimizers])
    assert all(parameter.grad is None for parameter in learner.agents.parameters())
    assert result['ppo_update_count'] == 0 and result['alive_fraction'] == 0
    assert result['sequential_factor_min'] == result['sequential_factor_max'] == 1
    assert result['sequential_factor_ess'] == 4


def test_correction_uses_cached_history_inputs_after_epochs_without_mutation(monkeypatch):
    learner = model(mode='sequential', history=3, algorithm='dms')
    data = batch(learner)
    saved = deepcopy(data)
    agent = learner.agents[0]
    original = agent.distribution
    calls = []
    def distribution(obs, context, gates, available=None):
        assert not torch.is_grad_enabled()
        calls.append((obs.clone(), context.clone(), gates.clone(), available.clone()))
        return original(obs, context, gates, available)
    monkeypatch.setattr(agent, 'distribution', distribution)
    def forbidden(*args, **kwargs):
        raise AssertionError('Correction must not renormalize observations or recompute intentions/gates')
    monkeypatch.setattr(agent, 'normalize_obs', forbidden)
    monkeypatch.setattr(agent, 'relevance', forbidden)
    monkeypatch.setattr(learner, 'contexts', forbidden)
    monkeypatch.setattr(learner.intentions[0], 'features', forbidden)
    with torch.no_grad():
        agent.obs_moments.mean.fill_(1234)
    correction = advance_sequential_correction(learner, data, 0, torch.zeros(4, requires_grad=True))
    assert not correction.requires_grad
    assert_same(calls[0][0], data['obs'][:, :, 0].flatten(0, 1))
    assert_same(calls[0][1], data['context'][:, :, 0].flatten(0, 1))
    assert_same(calls[0][2], data['gates'][:, :, 0].flatten(0, 1))
    assert_same(calls[0][3], data['available'][:, :, 0].flatten(0, 1))
    assert_same(data, saved)


def test_auxiliary_update_follows_all_completed_agent_epochs_and_preserves_rollout(monkeypatch):
    learner = model(mode='sequential', algorithm='dms')
    learner.config.epochs = 2
    data = batch(learner)
    saved = deepcopy(data)
    events = []
    actual = training.advance_sequential_correction
    def correction(learner, data, i, factor):
        # Each agent has processed both epochs before its likelihood refresh.
        assert all(state['step'].item() == 2 for state in learner.optimizers[i].state.values())
        events.append(i)
        return actual(learner, data, i, factor)
    def auxiliary(learner, data):
        assert sorted(events) == [0, 1]
        events.append('aux')
        return {}
    monkeypatch.setattr(training, 'advance_sequential_correction', correction)
    monkeypatch.setattr(training, 'update_intentions', auxiliary)
    result = update(learner, data)
    assert events == result['sequential_agent_order']+['aux']
    assert_same(data, saved)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 1000., -1000.])
def test_nonfinite_or_unrepresentable_correction_fails_without_clipping(value):
    with pytest.raises(FloatingPointError, match='sequential PPO correction'):
        sequential_factor(torch.tensor([0., value]))


def test_nonfinite_live_policy_ratio_fails_before_next_agent():
    learner = model(mode='sequential')
    data = batch(learner)
    data['log_prob'][0, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='sequential PPO correction'):
        advance_sequential_correction(learner, data, 0, torch.zeros(4))


@pytest.mark.parametrize('mode', ['independent', 'sequential'])
def test_only_sequential_draws_a_single_agent_permutation(monkeypatch, mode):
    learner = model(mode=mode)
    data = batch(learner)
    original = torch.randperm
    calls = []
    def permutation(n, device=None):
        result = original(n, device=device)
        calls.append((n, result.tolist()))
        return result
    monkeypatch.setattr(torch, 'randperm', permutation)
    result = update(learner, data)
    assert [n for n, _ in calls] == ([2, 4, 4] if mode == 'sequential' else [4, 4])
    if mode == 'sequential':
        assert result['sequential_agent_order'] == calls[0][1]


@pytest.mark.parametrize('algorithm', ['ippo', 'dms'])
def test_sequential_toy_run_with_policy_history_records_mode_and_training_signal(tmp_path, algorithm):
    output = tmp_path/algorithm
    config = Config(algorithm=algorithm, update_mode='sequential', output=str(output),
                    num_envs=2, rollout_steps=4, total_steps=8, hidden_dim=8, latent_dim=2,
                    policy_history_length=3, history_length=4, prediction_horizon=2,
                    epochs=2, minibatch_size=4, aux_updates=1, aux_batch_size=2,
                    eval_every=8, eval_episodes=1)
    training.run(config)
    manifest = json.loads((output/'manifest.json').read_text())
    metrics = json.loads((output/'metrics.jsonl').read_text().splitlines()[0])
    checkpoint = torch.load(output/'latest.pt', weights_only=False)
    assert checkpoint['config']['update_mode'] == 'sequential'
    assert checkpoint['steps'] == 8
    assert manifest['ppo_update']['mode'] == 'sequential'
    assert 'predecessor action likelihood ratios' in manifest['ppo_update']['training_signal']
    assert 'not the exact legacy' in manifest['ppo_update']['sequential_scope']
    assert sorted(metrics['sequential_agent_order']) == [0, 1, 2]
    assert 0 < metrics['sequential_factor_min'] <= metrics['sequential_factor_max']
    assert 0 < metrics['sequential_factor_ess_fraction'] <= 1
    if algorithm == 'dms':
        assert np.isfinite(metrics['intention_loss'])
        assert all(optimizer['state'] for optimizer in checkpoint['aux_optimizers'])
