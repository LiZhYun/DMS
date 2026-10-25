"""Pure graph checks run normally; native checks require an allocated opt-in."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from dms.legacy_env import LegacyMaMuJoCoEnv
from dms.legacy_graph import distance_layers, graph_for, local_observation, observation_categories


TASKS = [
    ("Ant", "2x4"), ("Ant", "4x2"), ("Ant", "8x1"),
    ("HalfCheetah", "2x3"), ("HalfCheetah", "3x2"), ("HalfCheetah", "6x1"),
    ("Walker2d", "2x3"), ("Walker2d", "3x2"), ("Walker2d", "6x1"),
]
native_only = pytest.mark.skipif(
    os.environ.get("DMS_TEST_LEGACY") != "1" or not os.environ.get("SLURM_JOB_ID"),
    reason="legacy native checks require DMS_TEST_LEGACY=1 in a Slurm allocation",
)


@pytest.mark.parametrize("scenario,partition", TASKS)
def test_original_partition_order_and_actuator_ownership(scenario, partition):
    parts, _ = graph_for(scenario, partition)
    size = int(partition.split("x")[1])
    expected = [2, 3, 4, 5, 6, 7, 0, 1] if scenario == "Ant" else [2, 1, 0, 5, 4, 3]
    assert [[joint.act_ids for joint in part] for part in parts] == [expected[start:start + size] for start in range(0, len(expected), size)]
    assert sorted(joint.act_ids for part in parts for joint in part) == list(range(len(expected)))


def test_original_halfcheetah_observation_order_and_neighbor_features():
    parts, edges = graph_for("HalfCheetah", "2x3")
    layers = distance_layers(parts[0], edges, 1)
    assert {radius: [joint.label for joint in joints] for radius, joints in layers.items()} == {
        0: ["bfoot", "bshin", "bthigh"], 1: ["fthigh"],
    }
    data = SimpleNamespace(qpos=np.arange(9.0), qvel=np.arange(9.0) + 10)
    np.testing.assert_array_equal(local_observation(data, layers, observation_categories("HalfCheetah", 1)), [5, 15, 4, 14, 3, 13, 6])


def test_ant_contact_clipping_deduplication_and_category_order():
    parts, edges = graph_for("Ant", "4x2")
    layers = distance_layers(parts[0], edges, 1)
    assert [joint.label for joint in layers[0]] == ["ankle1", "hip1"]
    assert [joint.label for joint in layers[1]] == ["hip2", "hip3", "hip4"]
    contacts = np.repeat(np.linspace(-1.4, 1.2, 14)[:, None], 6, axis=1)
    data = SimpleNamespace(qpos=np.arange(15.0), qvel=np.arange(14.0) + 20, cfrc_ext=contacts)
    # ankle1 first contributes bodies 2,3,4; hip1 then contributes only torso1.
    expected = [8, 27, *np.clip(contacts[2:5], -1, 1).ravel(), 7, 26, *np.clip(contacts[1], -1, 1), 9, 11, 13]
    actual = local_observation(data, layers, observation_categories("Ant", 1))
    np.testing.assert_allclose(actual, expected)
    assert len(actual) == 31


def test_walker_graph_velocity_is_not_single_agent_clipped_velocity():
    parts, edges = graph_for("Walker2d", "6x1")
    data = SimpleNamespace(qpos=np.arange(9.0), qvel=np.full(9, 25.0))
    actual = local_observation(data, distance_layers(parts[0], edges, 0), observation_categories("Walker2d", 0))
    np.testing.assert_array_equal(actual, [5, 25])


def test_graph_rejects_unknown_task_and_invalid_radius_before_native_import():
    with pytest.raises(ValueError, match="scenario"):
        graph_for("Hopper", "3x1")
    with pytest.raises(ValueError, match="partition"):
        graph_for("HalfCheetah", "2x4")
    with pytest.raises(ValueError, match="agent_obsk"):
        LegacyMaMuJoCoEnv(agent_obsk=-1)
    with pytest.raises(ValueError, match="max_episode_steps"):
        LegacyMaMuJoCoEnv(max_episode_steps=0)


@pytest.fixture
def legacy_factory():
    pytest.importorskip("gym", minversion="0.23.1")
    pytest.importorskip("mujoco_py")
    return LegacyMaMuJoCoEnv


@native_only
@pytest.mark.parametrize("scenario,partition", TASKS)
@pytest.mark.parametrize("radius", [0, 1])
def test_all_nine_native_tasks_seed_controls_sensors_and_timeout(legacy_factory, scenario, partition, radius):
    env = legacy_factory(scenario, partition, agent_obsk=radius, max_episode_steps=3)
    try:
        initial = env.reset(seed=17)
        np.testing.assert_array_equal(initial.obs, env.reset(seed=17).obs)
        assert initial.obs.shape == (env.n_agents, env.obs_dim)
        assert np.isfinite(initial.obs).all()
        assert not initial.visibility.any()
        assert not initial.observed_actions.any()
        commands = np.linspace(-0.2, 0.2, sum(env.action_dims), dtype=np.float32).reshape(env.n_agents, env.action_dim)
        transition = env.step(commands)
        expected_ctrl = np.zeros(sum(env.action_dims), dtype=np.float32)
        for agent, indices in enumerate(env.action_ids):
            expected_ctrl[indices] = commands[agent]
        np.testing.assert_allclose(env.native.sim.data.ctrl, expected_ctrl, rtol=0, atol=0)
        np.testing.assert_array_equal(transition.observed_actions, np.where(env.action_visibility, commands[None], 0))
        np.testing.assert_array_equal(transition.visibility, env.action_visibility.any(axis=-1))
        if radius == 0:
            np.testing.assert_array_equal(transition.visibility, np.eye(env.n_agents, dtype=bool))
        else:
            assert np.any(transition.visibility & ~np.eye(env.n_agents, dtype=bool))
        assert np.isfinite(transition.obs).all() and np.isfinite(transition.rewards).all()
        assert not transition.episode_done
        env.step(np.zeros_like(commands))
        final = env.step(np.zeros_like(commands))
        assert final.truncated.all() and not final.terminated.any()
        assert final.info["TimeLimit.truncated"]
        assert final.info["elapsed_steps"] == 3
        assert final.visibility.any()  # Final observation was not auto-reset.
        with pytest.raises(RuntimeError, match="reset"):
            env.step(commands)
        metadata = env.metadata
        assert metadata["mujoco_engine_version"] == 210
        assert metadata["packages"]["gym"] == "0.23.1"
        assert metadata["global_categories"] == []
        assert metadata["historical_binary_parity_verified"] is False
        assert len(metadata["xml_sha256"]) == len(metadata["adapted_graph_sha256"]) == 64
        assert metadata["action_ids"] == env.action_ids
    finally:
        env.close()


@native_only
@pytest.mark.parametrize("scenario,partition", [("Ant", "2x4"), ("Walker2d", "2x3")])
def test_true_terminal_at_time_limit_is_not_bootstrapped(legacy_factory, scenario, partition):
    env = legacy_factory(scenario, partition, max_episode_steps=1)
    try:
        env.reset(seed=17)
        qpos = env.native.sim.data.qpos.copy()
        height_index = 2 if scenario == "Ant" else 1
        # Starting inside the floor can launch Ant back into its healthy range
        # during one frame-skip. Start above the upper health bound instead.
        qpos[height_index] = 3.0
        env.native.set_state(qpos, np.zeros_like(env.native.sim.data.qvel))
        transition = env.step(np.zeros((env.n_agents, env.action_dim), dtype=np.float32))
        assert env.native.sim.data.qpos[height_index] > (1.0 if scenario == "Ant" else 2.0)
        assert transition.terminated.all() and not transition.truncated.any()
        assert not transition.info["TimeLimit.truncated"]
    finally:
        env.close()


@native_only
def test_clipped_executed_control_and_input_validation(legacy_factory):
    env = legacy_factory("HalfCheetah", "3x2")
    try:
        env.reset(seed=17)
        commands = np.array([[10, -10], [2, -2], [3, -3]], dtype=np.float32)
        original = commands.copy()
        transition = env.step(commands)
        np.testing.assert_array_equal(commands, original)
        np.testing.assert_array_equal(transition.observed_actions, np.where(env.action_visibility, np.clip(commands, -1, 1)[None], 0))
        with pytest.raises(ValueError, match="shape"):
            env.step(np.zeros(6))
        with pytest.raises(ValueError, match="finite"):
            env.step(np.full_like(commands, np.nan))
    finally:
        env.close()


@native_only
def test_legacy_backend_collect_and_update_has_finite_active_peer_inference(legacy_factory):
    import torch
    from dms.train import Config, Learner, collect, create_env, empty_history, seed_everything, update

    config = Config(
        env="mamujoco", mamujoco_backend="legacy", scenario="HalfCheetah", agent_conf="3x2",
        agent_obsk=1, num_envs=1, rollout_steps=4, total_steps=4, prediction_horizon=2,
        hidden_dim=8, latent_dim=2, history_length=4, short_history=2,
        epochs=1, minibatch_size=4, aux_updates=1, aux_batch_size=2, device="cpu",
    )
    config.validate()
    torch.set_num_threads(1)
    seed_everything(config.seed)
    env = create_env(config)
    try:
        learner = Learner(config, env)
        states = [env.reset(seed=config.seed)]
        history, visible = empty_history(learner, 1)
        data, *_ = collect(learner, [env], states, history, visible, 0, np.zeros(1))
        metrics = update(learner, data)
        assert all(np.isfinite(value) for value in metrics.values())
        assert metrics["peer_gate_fraction"] > 0
        assert metrics["intention_feature_std"] > 0
        assert "intention_loss" in metrics
    finally:
        env.close()
