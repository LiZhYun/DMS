import numpy as np
import pytest

from dms.envs import EnvStep
from dms.observation_history import ObservationHistoryEnv


class LocalEnvironment:
    n_agents, obs_dim, action_dim = 2, 2, 3
    metadata = {'environment': 'local_test'}
    runtime_info = {'restarts': 3}

    def __init__(self):
        self.time = 0
        self.dead = False
        self.closed = False
        self.peer_offset = 0

    def snapshot(self):
        return EnvStep(
            obs=np.array([[self.time, 10], [self.time + self.peer_offset, 20]], np.float32),
            rewards=np.ones(2, np.float32), terminated=np.zeros(2, bool),
            truncated=np.full(2, self.time == 2),
            action_mask=np.array([[True, not self.dead, not self.dead], [False, True, True]]),
            observed_actions=np.zeros((2, 2, 3), np.float32),
            visibility=np.eye(2, dtype=bool), info={'time': self.time})

    def reset(self, seed=None):
        self.time = 0 if seed is None else seed
        return self.snapshot()

    def step(self, actions):
        self.time += 1
        return self.snapshot()

    def close(self):
        self.closed = True


def test_history_ends_at_current_observation_and_old_snapshots_do_not_alias():
    env = ObservationHistoryEnv(LocalEnvironment(), 3)
    initial = env.reset()
    first = env.step(None)
    frozen_first = first.obs.copy()
    second = env.step(None)
    np.testing.assert_array_equal(initial.obs[0], [[0, 0, 0], [0, 0, 0], [0, 10, 1]])
    np.testing.assert_array_equal(first.obs[0], [[0, 0, 0], [0, 10, 1], [1, 10, 1]])
    np.testing.assert_array_equal(second.obs[0], [[0, 10, 1], [1, 10, 1], [2, 10, 1]])
    np.testing.assert_array_equal(first.obs, frozen_first)
    assert not np.shares_memory(first.obs, second.obs)
    assert second.episode_done  # The final window remains available for bootstrap.
    reset = env.reset(seed=7)
    np.testing.assert_array_equal(reset.obs[0], [[0, 0, 0], [0, 0, 0], [7, 10, 1]])
    np.testing.assert_array_equal(second.obs[0, :, 0], [0, 1, 2])


def test_death_clears_only_own_history_and_revival_has_no_old_inputs():
    native = LocalEnvironment()
    env = ObservationHistoryEnv(native, 3, clear_dead_agents=True)
    env.reset()
    env.step(None)
    native.dead = True
    dead = env.step(None)
    assert not dead.obs[0].any()
    np.testing.assert_array_equal(dead.obs[1, :, 0], [0, 1, 2])
    native.dead = False
    revived = env.step(None)
    np.testing.assert_array_equal(revived.obs[0], [[0, 0, 0], [0, 0, 0], [3, 10, 1]])


def test_other_agents_changes_do_not_enter_own_window_or_change_sensor_fields():
    native = LocalEnvironment()
    env = ObservationHistoryEnv(native, 2)
    env.reset()
    native.peer_offset = 9999
    state = env.step(None)
    np.testing.assert_array_equal(state.obs[0], [[0, 10, 1], [1, 10, 1]])
    np.testing.assert_array_equal(state.visibility, np.eye(2, dtype=bool))
    assert state.info == {'time': 1}
    assert state.obs[1, -1, 0] == 10000
    assert env.obs_dim == 2
    assert env.runtime_info == {'restarts': 3}
    assert 'policy_observation_history' not in native.metadata
    assert env.metadata['policy_observation_history']['shape'] == [2, 2, 3]
    env.close()
    assert native.closed


@pytest.mark.parametrize('length', [0, -1, 2.5, True])
def test_invalid_history_lengths_fail(length):
    with pytest.raises(ValueError, match='positive integer'):
        ObservationHistoryEnv(LocalEnvironment(), length)
