"""Own-action accuracy cannot establish teammate prediction quality."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import torch

from dms.envs import CoordinationToy
from dms.train import Config, Learner


spec = importlib.util.spec_from_file_location(
    'peer_forecast_probe', Path(__file__).resolve().parents[1] / 'scripts/diagnose_intentions.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def fixture():
    config = Config(num_envs=1, hidden_dim=8, latent_dim=2, history_length=4,
                    prediction_horizon=2, filter_history=False, warmup_episodes=0)
    env = CoordinationToy(n_agents=2)
    model = Learner(config, env)
    for intention in model.intentions:
        with torch.no_grad():
            intention.decoder_head.weight.zero_()
            intention.decoder_head.bias.copy_(torch.tensor([50., -50.]))
    history = np.zeros((4, 1, 4, 2, 2, 2), np.float32)
    history[..., 0] = 1
    visible = np.ones(history.shape[:-1], bool)
    with torch.no_grad():
        context = torch.stack([
            intention.features(torch.from_numpy(history[:, 0, :, i]),
                               torch.from_numpy(visible[:, 0, :, i]))
            for i, intention in enumerate(model.intentions)], 1).unsqueeze(1)
    future = np.zeros((4, 1, 2, 2, 2), np.float32)
    future[..., 0] = 1
    data = dict(ends=np.zeros((4, 1), bool), history=history, visible=visible,
                context=context, gates=(~torch.eye(2, dtype=torch.bool)).expand(4, 1, 2, 2).clone(),
                obs=torch.zeros(4, 1, 2, env.obs_dim),
                available=torch.ones(4, 1, 2, 2, dtype=torch.bool), future=future,
                future_visible=np.ones((4, 1, 2, 2), bool))
    return model, data


def test_peer_forecast_ignores_changed_self_targets_and_preserves_input_visibility():
    model, data = fixture()
    before_visible = data['future_visible'].copy()
    baseline = probe.diagnose(model, data, seed=101, max_windows=3)
    changed = deepcopy(data)
    for observer in range(2):
        changed['future'][:, :, observer, observer] = (0., 1.)
    altered = probe.diagnose(model, changed, seed=101, max_windows=3)
    assert baseline['forecast']['forecast_accuracy'] == 1
    assert altered['forecast']['forecast_accuracy'] == .5
    assert baseline['forecast_peer_only'] == altered['forecast_peer_only']
    assert baseline['forecast_peer_only']['forecast_accuracy'] == 1
    assert baseline['forecast_peer_only']['forecast_target_count'] == 12
    np.testing.assert_array_equal(data['future_visible'], before_visible)
    assert all(p.grad is None for p in model.intentions.parameters())


def test_self_only_targets_produce_no_peer_prediction_claim():
    model, data = fixture()
    data['future_visible'].fill(False)
    for observer in range(2):
        data['future_visible'][:, :, observer, observer] = True
    result = probe.diagnose(model, data, seed=101, max_windows=3)
    assert result['forecast']['forecast_target_count'] == 12
    assert result['forecast_peer_only'] == {}
