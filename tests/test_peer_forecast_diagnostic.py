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


def test_smac_gate_visibility_distinguishes_dead_observers_peers_and_sensor_age():
    # Observer1 is dead. Observer0 remembers dead peer1 at age2; observer2's
    # gate for peer1 was never observed. Both living observers see each other.
    gates = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    gates[0, 0, 1, 2] = False
    visible = np.zeros((1, 1, 3, 3, 3), bool)
    visible[0, 0, 0, 0, 1] = True
    visible[0, 0, 0, 1, 0] = True
    visible[0, 0, 2, 0, 2] = visible[0, 0, 2, 2, 0] = True
    available = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    available[0, 0, [0, 2], 1] = True
    available[0, 0, 1, 0] = True
    data = dict(gates=gates, visible=visible, available=available)
    before = deepcopy(data)
    result = probe.smac_peer_visibility(data)
    assert result['gated_peer_entries'] == 5  # Ignore deliberately true self gates.
    assert result['alive_observer_gated_peer_entries'] == 4
    assert result['latest_sensor_missing_gated_entries'] == 3
    assert result['latest_sensor_missing_alive_observer_gated_entries'] == 2
    assert result['dead_peer_alive_observer_gated_entries'] == 2
    assert result['latest_sensor_missing_alive_observer_gate_fraction'] == .5
    assert result['dead_peer_alive_observer_gate_fraction'] == .5
    assert result['never_sensed_alive_observer_gated_entries'] == 1
    assert result['alive_observer_gate_age_histogram'] == {'0': 2, '1': 0, '2': 1}
    torch.testing.assert_close(data['gates'], before['gates'])
    torch.testing.assert_close(data['available'], before['available'])
    np.testing.assert_array_equal(data['visible'], before['visible'])


def test_smac_empty_peer_gates_report_zero_counts_and_explicit_zero_denominator():
    data = dict(gates=torch.zeros(2, 1, 2, 2, dtype=torch.bool),
                visible=np.ones((2, 1, 3, 2, 2), bool),
                available=torch.ones(2, 1, 2, 2, dtype=torch.bool))
    result = probe.smac_peer_visibility(data)
    assert result['alive_observer_gated_peer_entries'] == 0
    assert result['latest_sensor_missing_alive_observer_gate_fraction'] == 0
    assert result['dead_peer_alive_observer_gate_fraction'] == 0
    assert result['alive_observer_gate_age_histogram'] == {'0': 0, '1': 0, '2': 0}
