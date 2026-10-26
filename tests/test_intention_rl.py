"""Joint RL/ELBO training must preserve local information and PPO ratios."""
from copy import deepcopy

import numpy as np
import pytest
import torch

import dms.train as training
from dms.train import Config, advance_sequential_correction, prepare_intention_inputs, update
from test_sequential_update import assert_same, batch, model


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def joint_batch(learner):
    data = batch(learner)
    h, n, a = learner.config.history_length, learner.num_agents, learner.action_dim
    rng = np.random.default_rng(31)
    data['history'] = rng.normal(size=(2, 2, h, n, n, a)).astype(np.float32)
    data['visible'] = rng.random((2, 2, h, n, n)) > .2
    data['future'] = np.full((2, 2, n, n, a), 777, np.float32)
    for i in range(n):
        past, visible = prepare_intention_inputs(learner, data, i)
        with torch.no_grad():
            data['context'][:, :, i] = learner.intentions[i].features(past, visible).reshape(2, 2, n, -1)
    with torch.no_grad():
        actions, log_prob, values = learner.action(
            data['obs'].flatten(0, 1), data['context'].flatten(0, 1),
            data['gates'].flatten(0, 1),
            data['available'].flatten(0, 1) if learner.discrete else None)
    for key, value in [('actions', actions), ('log_prob', log_prob), ('values', values)]:
        data[key] = value.reshape(2, 2, *value.shape[1:])
    return data


def joint_model(**settings):
    return model(algorithm='dms', intention_rl_lr=3e-4, history_length=4,
                 prediction_horizon=2, **settings)


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, '0.1'])
def test_invalid_joint_rate_is_rejected(value):
    with pytest.raises(ValueError, match='intention_rl_lr'):
        Config(intention_rl_lr=value).validate()


def test_joint_causal_selection_requires_separate_validation():
    with pytest.raises(ValueError, match='soft or all'):
        Config(intention_rl_lr=1e-4, coordination='causal').validate()
    Config(intention_rl_lr=0, coordination='causal').validate()
    # Like auxiliary settings, this field has no effect in IPPO.
    Config(algorithm='ippo', intention_rl_lr=1e-4, coordination='causal').validate()


@pytest.mark.parametrize('discrete,coordination', [(True, 'soft'), (False, 'soft'), (False, 'all')])
def test_reencoding_recovers_behavior_likelihood_before_optimization(discrete, coordination):
    learner = joint_model(discrete=discrete, coordination=coordination, intention_threshold=0)
    data = joint_batch(learner)
    for i in range(learner.num_agents):
        inputs = prepare_intention_inputs(learner, data, i)
        correction = advance_sequential_correction(learner, data, i, torch.zeros(4), inputs)
        torch.testing.assert_close(correction, torch.zeros(4), rtol=0, atol=0)


@pytest.mark.parametrize('coordination', ['soft', 'all'])
def test_rl_updates_prior_but_not_decoder_or_auxiliary_optimizer(coordination):
    learner = joint_model(coordination=coordination)
    data = joint_batch(learner)
    before = deepcopy(learner.intentions.state_dict())
    update(learner, data)
    after = learner.intentions.state_dict()
    assert any(not torch.equal(v, after[k]) for k, v in before.items() if '.encoder.' in k)
    for k, value in before.items():
        if '.decoder_' in k:
            assert_same(value, after[k])
    assert all(not optimizer.state for optimizer in learner.aux_optimizers)
    for i, intention in enumerate(learner.intentions):
        assert all(p in learner.optimizers[i].state for p in intention.prior_parameters())
        assert all(p.grad is None for name, p in intention.named_parameters() if name.startswith('decoder_'))


@pytest.mark.parametrize('mode', ['independent', 'sequential'])
def test_unobserved_actions_and_future_targets_do_not_change_ppo(mode):
    a = joint_model(mode=mode)
    data = joint_batch(a)
    b, changed = deepcopy(a), deepcopy(data)
    changed['history'][~changed['visible']] = float('nan')
    changed['future'][:] = float('nan')
    for learner, samples in [(a, data), (b, changed)]:
        torch.manual_seed(53)
        update(learner, samples)
    assert_same(a.agents.state_dict(), b.agents.state_dict())
    assert_same(a.intentions.state_dict(), b.intentions.state_dict())
    assert_same([o.state_dict() for o in a.optimizers], [o.state_dict() for o in b.optimizers])


def test_sequential_correction_includes_updated_encoder_with_original_filter():
    learner = joint_model(mode='sequential', intention_threshold=0)
    data = joint_batch(learner)
    inputs = prepare_intention_inputs(learner, data, 0)
    with torch.no_grad():
        learner.intentions[0].gaussian_head.bias[:learner.latent_dim].add_(2)
        context = learner.intentions[0].features(*inputs)
        dist = learner.agents[0].distribution(data['obs'][:, :, 0].flatten(0, 1), context,
            data['gates'][:, :, 0].flatten(0, 1), data['available'][:, :, 0].flatten(0, 1))
        expected = dist.log_prob(data['actions'][:, :, 0].flatten()) - data['log_prob'][:, :, 0].flatten()
    actual = advance_sequential_correction(learner, data, 0, torch.zeros(4), inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.abs().max() > 1e-5
    with pytest.raises(ValueError, match='original intention filter'):
        advance_sequential_correction(learner, data, 0, torch.zeros(4))


def test_no_peer_minibatches_do_not_advance_existing_prior_momentum():
    learner = joint_model()
    data = joint_batch(learner)
    update(learner, data)  # Create nonzero Adam moments first.
    prior = deepcopy(learner.intentions.state_dict())
    states = [{id(p): deepcopy(learner.optimizers[i].state[p]) for p in m.prior_parameters()}
              for i, m in enumerate(learner.intentions)]
    data['gates'].zero_()
    update(learner, data)
    assert_same(prior, learner.intentions.state_dict())
    for i, m in enumerate(learner.intentions):
        for p in m.prior_parameters():
            assert_same(states[i][id(p)], learner.optimizers[i].state[p])


def test_all_inactive_observer_does_not_advance_prior_or_policy():
    learner = joint_model(env='smac', mask=True)
    data = joint_batch(learner)
    update(learner, data)
    before = deepcopy(learner)
    data['available'].zero_()
    data['available'][..., 0] = True
    update(learner, data)
    assert_same(before.agents.state_dict(), learner.agents.state_dict())
    assert_same(before.intentions.state_dict(), learner.intentions.state_dict())
    assert_same([o.state_dict() for o in before.optimizers], [o.state_dict() for o in learner.optimizers])


def test_default_has_no_rl_intention_parameter_group_or_reencoding():
    learner = model(algorithm='dms')
    before = deepcopy(learner.intentions.state_dict())
    data = batch(learner)  # Deliberately has no history/future arrays.
    update(learner, data)
    assert all(len(o.param_groups) == 2 for o in learner.optimizers)
    assert_same(before, learner.intentions.state_dict())


@pytest.mark.parametrize('objective', ['actor', 'value'])
def test_each_rl_objective_can_train_the_prior(objective, monkeypatch):
    learner = joint_model(entropy_coef=0, value_coef=0 if objective == 'actor' else .5)
    data = joint_batch(learner)
    if objective == 'value':
        original = training.generalized_advantage_estimate
        def zero_advantage(*args, **kwargs):
            advantage, returns = original(*args, **kwargs)
            return torch.zeros_like(advantage), returns
        monkeypatch.setattr(training, 'generalized_advantage_estimate', zero_advantage)
    before = deepcopy(learner.intentions.state_dict())
    update(learner, data)
    assert any(not torch.equal(before[k], v) for k, v in learner.intentions.state_dict().items()
               if '.encoder.' in k)


@pytest.mark.parametrize('threshold', [0, 1e4])
def test_collected_likelihoods_reencode_across_warmup_and_episode_reset(threshold):
    config = Config(num_envs=1, rollout_steps=40, total_steps=40, hidden_dim=8,
                    latent_dim=2, history_length=4, prediction_horizon=2,
                    intention_rl_lr=3e-4, intention_threshold=threshold,
                    warmup_episodes=1, minibatch_size=7)
    env = training.create_env(config)
    try:
        learner = training.Learner(config, env)
        history, visible = training.empty_history(learner, 1)
        data, *_ = training.collect(learner, [env], [env.reset(seed=19)], history,
                                    visible, 0, np.zeros(1))
        assert not data['gates'][:32].any() and data['gates'][33:].any()
        for i in range(learner.num_agents):
            inputs = prepare_intention_inputs(learner, data, i)
            ratio = advance_sequential_correction(learner, data, i, torch.zeros(40), inputs)
            torch.testing.assert_close(ratio, torch.zeros(40), rtol=0, atol=2e-7)
    finally:
        env.close()
