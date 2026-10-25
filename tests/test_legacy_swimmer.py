"""10x2 graph/asset checks are pure; native physics requires allocated opt-in."""

import hashlib
import os
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from dms.envs import _joint_action_visibility
from dms.legacy_env import LegacyMaMuJoCoEnv
from dms.legacy_graph import distance_layers, graph_for, local_observation, observation_categories
from dms.legacy_swimmer import ASSET_PATH


native_only = pytest.mark.skipif(
    os.environ.get("DMS_TEST_LEGACY") != "1" or not os.environ.get("SLURM_JOB_ID"),
    reason="legacy native checks require DMS_TEST_LEGACY=1 in a Slurm allocation",
)


def test_packaged_asset_matches_inspected_reference_generator():
    # The fingerprint is of the independently reconstructed reference rendering,
    # before adding the retained source/license comment. No reference checkout,
    # templating package, or simulator is needed to check the shipped asset.
    contents = ASSET_PATH.read_bytes()
    rendered = contents.split(b"-->\n", 1)[1]
    assert hashlib.sha256(rendered).hexdigest() == "636870e46fad00f8cc9957150c2e706c0caba5cb9b4a06ae8a12fe143b8ce388"
    root = ET.fromstring(contents)
    assert root.find("option").attrib == {
        "collision": "predefined", "density": "4000", "integrator": "RK4",
        "timestep": "0.005", "viscosity": "0.1",
    }
    motors = root.findall("./actuator/motor")
    assert len(motors) == 20
    assert [motor.attrib["joint"] for motor in motors] == [f"rot{i}" for i in range(20)]
    assert all(motor.attrib["gear"] == "150.0" and motor.attrib["ctrlrange"] == "-1 1" for motor in motors)
    assert len(root.findall(".//worldbody//joint")) == 23  # two slides, root yaw, 20 controls


def test_swimmer_graph_routes_consecutive_pairs_along_one_chain():
    parts, edges = graph_for("ManyAgentSwimmer", "10x2")
    assert [[joint.act_ids for joint in part] for part in parts] == [[2*i, 2*i+1] for i in range(10)]
    assert [(joint.qpos_ids, joint.qvel_ids) for part in parts for joint in part] == [(i-20, i-20) for i in range(20)]
    assert {frozenset(joint.act_ids for joint in edge) for edge in edges} == {frozenset((i,i+1)) for i in range(19)}


@pytest.mark.parametrize("radius", [0, 1])
def test_swimmer_local_sensor_has_only_own_or_adjacent_joint_controls(radius):
    parts, edges = graph_for("ManyAgentSwimmer", "10x2")
    layers = [distance_layers(part, edges, radius) for part in parts]
    visibility = _joint_action_visibility(parts, layers, 2)
    expected = np.zeros((10, 10, 2), dtype=bool)
    for observer in range(10):
        expected[observer, observer] = True
        if radius and observer > 0:
            expected[observer, observer-1, 1] = True
        if radius and observer < 9:
            expected[observer, observer+1, 0] = True
    np.testing.assert_array_equal(visibility, expected)
    data = SimpleNamespace(qpos=np.arange(23.0), qvel=100+np.arange(23.0))
    categories = observation_categories("ManyAgentSwimmer", radius)
    baseline = local_observation(data, layers[5], categories)
    expected_values = [13, 113, 14, 114, 15, 12] if radius else [13, 113, 14, 114]
    np.testing.assert_array_equal(baseline, expected_values)
    # Root position, root velocity and a distant limb are private, not globals.
    data.qpos[[0,1,2,3]] += 1000
    data.qvel[[0,1,2,3]] -= 1000
    np.testing.assert_array_equal(local_observation(data, layers[5], categories), baseline)


def test_swimmer_rejects_other_partitions_before_native_import():
    for partition in ("2x1", "20x1", "10x3"):
        with pytest.raises(ValueError, match="10x2"):
            LegacyMaMuJoCoEnv("ManyAgentSwimmer", partition)


@native_only
@pytest.mark.parametrize("radius", [0, 1])
def test_native_swimmer_seeded_reset_control_routing_and_local_history(radius):
    env = LegacyMaMuJoCoEnv("ManyAgentSwimmer", "10x2", radius, max_episode_steps=3)
    original_asset = ASSET_PATH.read_bytes()
    try:
        initial = env.reset(seed=17)
        np.testing.assert_array_equal(initial.obs, env.reset(seed=17).obs)
        rng = np.random.default_rng(17)
        np.testing.assert_allclose(env.native.sim.data.qpos, env.native.init_qpos+rng.uniform(-.1,.1,23), rtol=0, atol=1e-14)
        np.testing.assert_allclose(env.native.sim.data.qvel, env.native.init_qvel+rng.uniform(-.1,.1,23), rtol=0, atol=1e-14)
        assert initial.obs.shape == (10, env.obs_dim)
        assert env.observation_dims == ([4]*10 if radius == 0 else [5]+[6]*8+[5])
        assert not initial.visibility.any() and not initial.observed_actions.any()
        actions = np.linspace(-2, 2, 20, dtype=np.float32).reshape(10,2)
        transition = env.step(actions)
        clipped = np.clip(actions, -1, 1)
        np.testing.assert_array_equal(env.native.sim.data.ctrl, clipped.ravel())
        np.testing.assert_array_equal(transition.observed_actions, np.where(env.action_visibility, clipped[None], 0))
        np.testing.assert_array_equal(transition.visibility, env.action_visibility.any(-1))
        assert np.isfinite(transition.obs).all() and np.isfinite(transition.rewards).all()
        env.step(np.zeros_like(actions))
        final = env.step(np.zeros_like(actions))
        assert final.truncated.all() and not final.terminated.any()
        assert final.info["TimeLimit.truncated"] and final.info["elapsed_steps"] == 3
        assert final.visibility.any()  # terminal state was not auto-reset
        with pytest.raises(RuntimeError, match="reset"):
            env.step(actions)
        reset = env.reset(seed=17)
        np.testing.assert_array_equal(reset.obs, initial.obs)
        assert not reset.visibility.any() and not reset.observed_actions.any()
        assert ASSET_PATH.read_bytes() == original_asset
    finally:
        env.close()


@native_only
def test_swimmer_dynamics_and_reward_match_direct_mujoco_integration():
    import mujoco_py

    env = LegacyMaMuJoCoEnv("ManyAgentSwimmer", "10x2")
    try:
        env.reset(seed=31)
        direct = mujoco_py.MjSim(mujoco_py.load_model_from_path(str(ASSET_PATH.resolve())))
        direct.set_state(env.native.sim.get_state())
        direct.forward()
        assert env.native.model.nq == env.native.model.nv == 23
        assert env.native.model.nu == 20
        assert env.native.frame_skip == 4 and env.native.dt == pytest.approx(.02)
        assert env.native.model.opt.integrator == 1  # MuJoCo RK4
        assert env.native.model.opt.density == 4000 and env.native.model.opt.viscosity == .1
        for index in range(8):
            actions = (.15*np.sin(np.arange(20)+index)).astype(np.float32).reshape(10,2)
            before = direct.data.qpos[0]
            direct.data.ctrl[:] = actions.ravel()
            for _ in range(4):
                direct.step()
            expected_forward = (direct.data.qpos[0]-before)/.02
            expected_control = -.0001*np.square(actions.ravel()).sum()
            transition = env.step(actions)
            np.testing.assert_allclose(env.native.sim.data.qpos, direct.data.qpos, rtol=0, atol=1e-12)
            np.testing.assert_allclose(env.native.sim.data.qvel, direct.data.qvel, rtol=0, atol=1e-12)
            assert transition.info["reward_fwd"] == pytest.approx(expected_forward, abs=1e-12)
            assert transition.info["reward_ctrl"] == pytest.approx(expected_control, abs=1e-12)
            np.testing.assert_allclose(transition.rewards, expected_forward+expected_control, rtol=1e-6, atol=1e-7)
            assert not transition.terminated.any()
        metadata = env.metadata
        assert metadata["xml_sha256"] == hashlib.sha256(ASSET_PATH.read_bytes()).hexdigest()
        assert metadata["native_env_id"] == "custom_manyagent_swimmer_10x2"
        assert metadata["global_categories"] == [] and metadata["historical_binary_parity_verified"] is False
        assert len(metadata["original_environment_sha256"]) == len(metadata["original_xml_template_sha256"]) == 64
    finally:
        env.close()


@native_only
def test_swimmer_default_episode_is_truncated_at_exactly_1000_steps():
    env = LegacyMaMuJoCoEnv("ManyAgentSwimmer", "10x2")
    try:
        env.reset(seed=23)
        actions = np.zeros((10,2), dtype=np.float32)
        for _ in range(999):
            assert not env.step(actions).episode_done
        final = env.step(actions)
        assert final.truncated.all() and not final.terminated.any()
        assert final.info["elapsed_steps"] == 1000
    finally:
        env.close()
