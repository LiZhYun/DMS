"""Inactive SMAC rows must not alter an opted-in PPO update."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dms.envs import CoordinationToy
from dms.train import Config, Learner, run, update


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def learner(*, mask=True, algorithm='ippo', env='smac', discrete=True):
    config = Config(algorithm=algorithm, env=env, num_envs=1,
                    rollout_steps=4, total_steps=4, epochs=2, minibatch_size=4,
                    hidden_dim=8, latent_dim=2, prediction_horizon=2,
                    aux_updates=0, target_kl=0.0, mask_inactive_agents=mask)
    torch.manual_seed(7)
    return Learner(config, SimpleNamespace(n_agents=2, action_dim=3,
                   action_dims=[3, 3], obs_dim=4, discrete=discrete))


def batch(model, alive):
    alive = torch.as_tensor(alive, dtype=torch.bool).reshape(-1, 1, 2)
    length = len(alive)
    raw = torch.arange(length*8, dtype=torch.float32).reshape(length, 1, 2, 4)/10
    raw = torch.where(alive[..., None], raw, raw+10000)
    obs = torch.stack([a.normalize_obs(raw[:, 0, i])
                       for i, a in enumerate(model.agents)], 1)[:, None]
    contexts = torch.zeros(length, 1, 2, 2, model.latent_dim)
    gates = torch.zeros(length, 1, 2, 2, dtype=torch.bool)
    available = torch.zeros(length, 1, 2, 3, dtype=torch.bool)
    available[..., 0] = ~alive
    available[..., 1:] = alive[..., None]
    with torch.no_grad():
        actions, log_prob, values = model.action(
            obs[:, 0], contexts[:, 0], gates[:, 0], available[:, 0] if model.discrete else None)
    result = dict(obs=obs.detach(), raw_obs=raw.numpy(), context=contexts, gates=gates,
                  actions=actions[:, None], log_prob=log_prob[:, None], values=values[:, None],
                  next_values=torch.zeros(length, 1, 2),
                  rewards=torch.where(alive, 1+raw[..., 0], 1e6).numpy(),
                  # Independent completed samples avoid changing live GAE suffixes
                  # when padding the batch with unrelated inactive observations.
                  terminated=np.ones((length, 1, 2), bool),
                  truncated=np.zeros((length, 1, 2), bool),
                  ends=np.ones((length, 1), bool))
    if model.discrete:
        result['available'] = available
    return result


def concatenate(a, b):
    return {key: (torch.cat([value, b[key]]) if isinstance(value, torch.Tensor)
                  else np.concatenate([value, b[key]])) for key, value in a.items()}


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal_state(left, right)
    else:
        assert a == b


@pytest.mark.parametrize('algorithm', ['ippo', 'dms'])
def test_added_dead_samples_cannot_change_live_update_or_normalizers(algorithm):
    a = learner(algorithm=algorithm)
    b = deepcopy(a)
    active = batch(a, [[True, True]]*4)
    padded = concatenate(active, batch(a, [[False, False]]*3))
    torch.manual_seed(19)
    clean_metrics = update(a, active)
    torch.manual_seed(19)
    padded_metrics = update(b, padded)
    equal_state(a.agents.state_dict(), b.agents.state_dict())
    equal_state([o.state_dict() for o in a.optimizers], [o.state_dict() for o in b.optimizers])
    for key in ['policy_loss', 'value_loss', 'entropy', 'approx_kl', 'ppo_update_count']:
        assert clean_metrics[key] == padded_metrics[key]
    assert clean_metrics['alive_fraction'] == 1
    assert padded_metrics['alive_fraction'] == pytest.approx(4/7)
    for agent in b.agents:
        assert agent.obs_moments.count.item() == pytest.approx(4.0001)
        assert agent.value_moments.count.item() == pytest.approx(4.0001)


def test_all_dead_after_adam_warmup_leaves_parameters_buffers_and_optimizer_unchanged():
    model = learner()
    update(model, batch(model, [[True, True]]*4))
    params = deepcopy(model.agents.state_dict())
    optimizer = deepcopy([o.state_dict() for o in model.optimizers])
    assert any(p.grad is not None for p in model.agents.parameters())
    metrics = update(model, batch(model, [[False, False]]*4))
    equal_state(params, model.agents.state_dict())
    equal_state(optimizer, [o.state_dict() for o in model.optimizers])
    assert all(p.grad is None for p in model.agents.parameters())
    assert metrics['alive_fraction'] == 0 and metrics['ppo_update_count'] == 0
    assert all(np.isfinite(value) for value in metrics.values())


def test_one_fully_dead_agent_is_skipped_while_the_other_updates():
    model = learner()
    original = deepcopy(model.agents.state_dict())
    metrics = update(model, batch(model, [[False, True]]*4))
    equal_state({k.removeprefix('0.'):v for k,v in original.items() if k.startswith('0.')},
                model.agents[0].state_dict())
    assert not model.optimizers[0].state
    assert model.optimizers[1].state
    assert metrics['alive_fraction'] == .5
    assert metrics['ppo_update_count'] == model.config.epochs
    assert model.agents[1].obs_moments.count.item() == pytest.approx(4.0001)


def test_dead_padding_cannot_dilute_the_actor_kl_stopping_rule():
    a = learner()
    a.config.target_kl = 1e-6
    a.config.epochs = 6
    a.config.minibatch_size = 2
    for optimizer in a.optimizers:
        optimizer.param_groups[0]['lr'] = .2
    b = deepcopy(a)
    active = batch(a, [[True, True]]*4)
    padded = concatenate(active, batch(a, [[False, False]]*12))
    torch.manual_seed(23)
    clean = update(a, active)
    torch.manual_seed(23)
    mixed = update(b, padded)
    assert clean['ppo_update_count'] < 2*a.config.epochs*2
    assert mixed['ppo_update_count'] == clean['ppo_update_count']
    assert mixed['approx_kl'] == clean['approx_kl']
    equal_state(a.agents.state_dict(), b.agents.state_dict())


@pytest.mark.parametrize('env,discrete', [('toy', True), ('mamujoco', False)])
def test_option_does_not_change_toy_or_continuous_updates(env, discrete):
    a = learner(mask=False, env=env, discrete=discrete)
    b = deepcopy(a)
    b.config.mask_inactive_agents = True
    data = batch(a, [[False, True], [True, False], [True, True], [False, False]])
    torch.manual_seed(37)
    before = update(a, data)
    torch.manual_seed(37)
    after = update(b, data)
    assert before == after
    equal_state(a.agents.state_dict(), b.agents.state_dict())
    equal_state([o.state_dict() for o in a.optimizers], [o.state_dict() for o in b.optimizers])


def test_default_remains_unmasked_and_still_logs_alive_fraction():
    assert Config().mask_inactive_agents is False
    model = learner(mask=False)
    metrics = update(model, batch(model, [[False, False]]*4))
    assert metrics['alive_fraction'] == 0
    assert metrics['ppo_update_count'] > 0
    assert all(o.state for o in model.optimizers)
    assert all(a.obs_moments.count.item() == pytest.approx(4.0001) for a in model.agents)


@pytest.mark.parametrize('mask', [False, True])
def test_training_battle_statistics_do_not_depend_on_mask_selection(tmp_path, monkeypatch, mask):
    class ToyBattle(CoordinationToy):
        runtime_info = {'scheduled_restarts': 0}

        def __init__(self):
            super().__init__(horizon=2)
            self.episode_index = 0

        def reset(self, seed=None):
            self.episode_index += 1
            return super().reset(seed)

        def step(self, actions):
            state = super().step(actions)
            return replace(state, info={**state.info, 'battle_won': self.episode_index % 2 == 1})

    import dms.train as training
    monkeypatch.setattr(training, 'create_env', lambda config: ToyBattle())
    config = Config(env='smac', algorithm='ippo', output=str(tmp_path/'run'),
                    mask_inactive_agents=mask, num_envs=1, rollout_steps=2, total_steps=4,
                    epochs=1, hidden_dim=8, prediction_horizon=1, aux_updates=0,
                    eval_every=4, eval_episodes=1)
    run(config)
    rows = [json.loads(x) for x in (tmp_path/'run/metrics.jsonl').read_text().splitlines()]
    assert [r['training_battles'] for r in rows] == [1, 1]
    assert [r['training_battles_won'] for r in rows] == [1, 0]
    assert [r['training_win_rate'] for r in rows] == [1.0, 0.0]
    assert [r['training_battles_total'] for r in rows] == [1, 2]
    assert [r['training_battles_won_total'] for r in rows] == [1, 1]
    assert [r['training_win_rate_total'] for r in rows] == [1.0, .5]
    assert all(r['alive_fraction'] == 1 for r in rows)
