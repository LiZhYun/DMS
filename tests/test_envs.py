from types import SimpleNamespace

import numpy as np
import pytest

from dms.envs import CoordinationToy, MaMuJoCoEnv, make_env


class _GraphBackend:
    """Small deterministic stand-in exercising partial peer visibility."""

    def __init__(self):
        joints = [object() for _ in range(5)]
        self.possible_agents = ["a0", "a1", "a2"]
        self.agent_action_partitions = [joints[:2], joints[2:4], joints[4:]]
        self.k_dicts = [
            {0: joints[:2], 1: [joints[2]]},
            {0: joints[2:4], 1: [joints[1], joints[4]]},
            {0: joints[4:], 1: [joints[3]]},
        ]
        self._action_dims = dict(zip(self.possible_agents, [2, 2, 1]))
        self._obs_dims = dict(zip(self.possible_agents, [2, 3, 1]))
        self.closed = False

    def action_space(self, agent):
        size = self._action_dims[agent]
        return SimpleNamespace(shape=(size,), low=np.full(size, -1.0), high=np.full(size, 1.0))

    def observation_space(self, agent):
        return SimpleNamespace(shape=(self._obs_dims[agent],))

    def _obs(self):
        return {agent: np.arange(size, dtype=np.float32) for agent, size in self._obs_dims.items()}

    def reset(self, seed=None):
        self.time = 0
        return self._obs(), {"seed": seed}

    def step(self, actions):
        self.last_actions = {agent: action.copy() for agent, action in actions.items()}
        self.time += 1
        return (
            self._obs(),
            dict.fromkeys(self.possible_agents, 2.5),
            dict.fromkeys(self.possible_agents, False),
            dict.fromkeys(self.possible_agents, self.time >= 2),
            {},
        )

    def close(self):
        self.closed = True


def test_local_torque_sensor_does_not_reveal_hidden_peer_components():
    env = MaMuJoCoEnv(backend=_GraphBackend())
    initial = env.reset(seed=2)
    assert not initial.visibility.any()
    assert not initial.observed_actions.any()
    assert env.action_dims == [2, 2, 1]
    np.testing.assert_array_equal(env.action_visibility[0, 1], [True, False])
    assert not env.action_visibility[0, 2].any()
    assert not env.action_visibility[:, 2, 1].any()  # Padding is never observed.

    first = env.step(np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 99.0]]))
    env.reset(seed=2)
    changed_hidden = env.step(np.array([[0.1, 0.2], [0.3, 0.9], [-0.8, -99.0]]))
    np.testing.assert_array_equal(first.observed_actions[0], changed_hidden.observed_actions[0])
    np.testing.assert_array_equal(first.obs[0], changed_hidden.obs[0])
    np.testing.assert_array_equal(first.visibility[0], [True, True, False])
    np.testing.assert_allclose(first.observed_actions[0, 1], [0.3, 0.0])
    assert first.obs.shape == (env.n_agents, env.obs_dim)


def test_continuous_adapter_clips_executed_commands_and_preserves_final_obs():
    backend = _GraphBackend()
    env = MaMuJoCoEnv(backend=backend)
    env.reset()
    commands = np.array([[10.0, -10.0], [2.0, -2.0], [3.0, 99.0]])
    transition = env.step(commands)
    np.testing.assert_array_equal(backend.last_actions["a0"], [1.0, -1.0])
    np.testing.assert_array_equal(backend.last_actions["a2"], [1.0])
    np.testing.assert_array_equal(transition.observed_actions[0, 0], [1.0, -1.0])
    np.testing.assert_array_equal(commands[0], [10.0, -10.0])  # PPO sample remains untouched.
    final = env.step(commands)
    assert final.episode_done
    assert final.truncated.all()
    assert not final.terminated.any()
    assert final.visibility.any()  # No hidden auto-reset cleared the final sensor.
    with pytest.raises(RuntimeError, match="reset"):
        env.step(commands)
    env.close()
    assert backend.closed


def test_toy_seed_history_visibility_and_timeout():
    env = CoordinationToy(n_agents=3, horizon=2)
    first_reset = env.reset(seed=4)
    np.testing.assert_array_equal(first_reset.obs[1:, :2], 0.0)
    assert first_reset.obs[0, :2].sum() == 1.0
    assert not first_reset.visibility.any()
    first = env.step(np.array([0, 1, 0]))
    np.testing.assert_array_equal(first.visibility, [[True, False, True], [True, True, False], [False, True, True]])
    np.testing.assert_array_equal(first.observed_actions[2, 0], [0.0, 0.0])
    np.testing.assert_array_equal(first.observed_actions[1, 0], [1.0, 0.0])
    assert first.action_mask.all()
    assert not first.episode_done
    final = env.step(np.array([1, 0, 1]))
    assert final.truncated.all() and not final.terminated.any()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(3, dtype=int))
    repeated = env.reset(seed=4)
    np.testing.assert_array_equal(repeated.obs, first_reset.obs)
    np.testing.assert_array_equal(env.step(np.array([0, 1, 0])).rewards, first.rewards)


def test_toy_rejects_fractional_or_invalid_actions():
    env = make_env("toy", horizon=2)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="integer"):
        env.step(np.array([0.5, 0.0, 1.0]))
    with pytest.raises(ValueError, match="outside"):
        env.step(np.array([0, 0, 2]))


def test_modern_mamujoco_real_reset_step_timeout_and_sparse_visibility():
    pytest.importorskip("gymnasium_robotics")
    pytest.importorskip("mujoco")
    env = MaMuJoCoEnv(scenario="HalfCheetah", agent_conf="6x1", agent_obsk=1, max_episode_steps=2)
    try:
        first = env.reset(seed=7)
        assert first.obs.shape == (6, env.obs_dim)
        assert np.isfinite(first.obs).all()
        assert not env.action_visibility.all()
        assert np.diag(env.action_visibility[:, :, 0]).all()
        transition = env.step(np.zeros((6, 1), dtype=np.float32))
        assert transition.rewards.shape == (6,)
        assert np.isfinite(transition.rewards).all()
        final = env.step(np.zeros((6, 1), dtype=np.float32))
        assert final.truncated.all() and not final.terminated.any()
        assert final.episode_done
        assert final.visibility.any()
    finally:
        env.close()


@pytest.mark.parametrize("agent_obsk", [0, 1])
@pytest.mark.parametrize(
    "scenario,agent_conf",
    [("Ant", "2x4"), ("Ant", "4x2"), ("Ant", "8x1")]
    + [(scenario, partition) for scenario in ("HalfCheetah", "Walker2d") for partition in ("2x3", "3x2", "6x1")],
)
def test_all_paper_partitions_and_observation_radii(scenario, agent_conf, agent_obsk):
    pytest.importorskip("gymnasium_robotics")
    pytest.importorskip("mujoco")
    env = MaMuJoCoEnv(scenario=scenario, agent_conf=agent_conf, agent_obsk=agent_obsk, max_episode_steps=2)
    try:
        agents, action_dims = map(int, agent_conf.split("x"))
        assert env.n_agents == agents
        assert env.action_dims == [action_dims] * agents
        assert tuple(env.backend.global_categories) == ()
        first = env.reset(seed=11)
        assert first.obs.shape == (agents, env.obs_dim)
        assert np.isfinite(first.obs).all()
        transition = env.step(np.zeros((agents, action_dims), dtype=np.float32))
        assert np.isfinite(transition.obs).all()
        assert np.isfinite(transition.rewards).all()
        if agent_obsk == 0:
            np.testing.assert_array_equal(transition.visibility, np.eye(agents, dtype=bool))
        else:
            assert np.any(transition.visibility & ~np.eye(agents, dtype=bool))
        if agent_conf == "3x2":
            actuator_groups = [[node.act_ids for node in partition] for partition in env.backend.agent_action_partitions]
            assert actuator_groups == [[2, 1], [0, 5], [4, 3]]
            # Commands reach their declared physical joints, irrespective of
            # the original wrapper's concatenation-order bug.
            commands = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
            env.step(commands)
            np.testing.assert_allclose(env.backend.single_agent_env.unwrapped.data.ctrl, [0.3, 0.2, 0.1, 0.6, 0.5, 0.4])
        if agent_conf == "8x1":
            assert [partition[0].act_ids for partition in env.backend.agent_action_partitions] == [2, 3, 4, 5, 6, 7, 0, 1]
    finally:
        env.close()
