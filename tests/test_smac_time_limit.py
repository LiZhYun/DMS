"""A finite battle deadline changes value targets, not observed transitions."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dms.train import Config, Learner, update


@pytest.mark.parametrize('mode', ['independent', 'sequential'])
@pytest.mark.parametrize('terminal', [False, True])
def test_deadline_targets_preserve_true_terminals_and_rollout_bootstrap(monkeypatch, mode, terminal):
    torch.set_num_threads(1)
    config = Config(algorithm='ippo', env='smac', hidden_dim=8, epochs=1,
                    minibatch_size=4, gamma=.9, gae_lambda=1, target_kl=0,
                    smac_time_limit_terminal=terminal, update_mode=mode)
    env = SimpleNamespace(n_agents=2, obs_dim=3, action_dim=3,
                          action_dims=[3, 3], discrete=True)
    torch.manual_seed(31)
    learner = Learner(config, env)
    observations = torch.randn(4, 2, 3)
    contexts = torch.zeros(4, 2, 2, 0)
    gates = torch.zeros(4, 2, 2, dtype=torch.bool)
    available = torch.ones(4, 2, 3, dtype=torch.bool)
    available[..., 0] = False
    with torch.no_grad():
        actions, log_prob, _ = learner.action(observations, contexts, gates, available)
    data = dict(obs=observations[:, None], raw_obs=observations[:, None].numpy(),
                context=contexts[:, None], gates=gates[:, None],
                actions=actions[:, None], log_prob=log_prob[:, None],
                available=available[:, None], values=torch.full((4, 1, 2), 10.),
                next_values=torch.full((4, 1, 2), 10.),
                rewards=np.arange(1, 9, dtype=np.float32).reshape(4, 1, 2),
                terminated=np.array([False, False, True, False])[:, None, None].repeat(2, 2),
                truncated=np.array([False, True, False, False])[:, None, None].repeat(2, 2))
    before = deepcopy(data)
    captured = {}
    for index, agent in enumerate(learner.agents):
        original = agent.update_value_scale
        def capture(targets, index=index, original=original):
            captured[index] = targets.clone()
            original(targets)
        monkeypatch.setattr(agent, 'update_value_scale', capture)
    update(learner, data)
    # Row 1 is a deadline; row 2 truly terminates; row 3 is only a batch boundary.
    expected = ([[3.7, 5.6], [3., 4.], [5., 6.], [16., 17.]] if terminal else
                [[11.8, 13.7], [12., 13.], [5., 6.], [16., 17.]])
    torch.testing.assert_close(torch.stack([captured[i] for i in range(2)], -1),
                               torch.tensor(expected), atol=2e-6, rtol=1e-6)
    for name, value in data.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        else:
            np.testing.assert_array_equal(value, before[name])


def test_deadline_option_is_explicit_and_smac_only():
    assert Config().smac_time_limit_terminal is False
    Config(env='smac', smac_time_limit_terminal=True).validate()
    for value in [0, 1, 'false', None]:
        with pytest.raises(ValueError, match='boolean'):
            Config(env='smac', smac_time_limit_terminal=value).validate()
    for env in ['toy', 'mamujoco']:
        with pytest.raises(ValueError, match='env=smac'):
            Config(env=env, smac_time_limit_terminal=True).validate()
