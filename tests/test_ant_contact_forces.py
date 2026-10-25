"""Explicit Ant force computation is a named, opt-in environment variant."""
import os

import numpy as np
import pytest

from dms.legacy_env import LegacyMaMuJoCoEnv


native_only = pytest.mark.skipif(
    os.environ.get("DMS_TEST_LEGACY") != "1" or not os.environ.get("SLURM_JOB_ID"),
    reason="legacy native checks require DMS_TEST_LEGACY=1 in a Slurm allocation",
)


@pytest.mark.parametrize("mode", ["", "automatic", "Computed"])
def test_invalid_force_mode_rejected_before_native_import(mode):
    with pytest.raises(ValueError, match="ant_contact_forces"):
        LegacyMaMuJoCoEnv("Ant", "4x2", ant_contact_forces=mode)


@pytest.mark.parametrize("scenario,partition", [("HalfCheetah", "2x3"), ("Walker2d", "3x2"), ("ManyAgentSwimmer", "10x2")])
def test_computed_forces_cannot_silently_select_another_scenario(scenario, partition):
    with pytest.raises(ValueError, match="only for legacy Ant"):
        LegacyMaMuJoCoEnv(scenario, partition, ant_contact_forces="computed")


@native_only
def test_native_default_matches_explicit_native_and_keeps_zero_force_behavior():
    pytest.importorskip("mujoco_py")
    default = LegacyMaMuJoCoEnv("Ant", "4x2")
    explicit = LegacyMaMuJoCoEnv("Ant", "4x2", ant_contact_forces="native")
    try:
        np.testing.assert_array_equal(default.reset(seed=191101).obs, explicit.reset(seed=191101).obs)
        contacts = 0
        for _ in range(30):
            command = np.zeros((4, 2), np.float32)
            a, b = default.step(command), explicit.step(command)
            np.testing.assert_array_equal(a.obs, b.obs)
            np.testing.assert_array_equal(a.rewards, b.rewards)
            np.testing.assert_array_equal(default.native.sim.data.qpos, explicit.native.sim.data.qpos)
            assert not default.native.sim.data.cfrc_ext.any()
            assert a.info["reward_contact"] == 0
            contacts += default.native.sim.data.ncon > 0
        assert contacts > 0
        assert default.metadata["ant_contact_forces"] == "native"
        assert default.metadata["benchmark_track"] == "historical_gym_v2_mujoco210_compatibility"
        assert default.metadata["contact_force_computation"]["api"] is None
    finally:
        default.close(); explicit.close()


@native_only
@pytest.mark.parametrize("partition", ["2x4", "4x2", "8x1"])
def test_computed_local_observations_and_reward_match_explicit_forces_without_changing_physics(partition):
    mujoco_py = pytest.importorskip("mujoco_py")
    native = LegacyMaMuJoCoEnv("Ant", partition)
    computed = LegacyMaMuJoCoEnv("Ant", partition, ant_contact_forces="computed")
    try:
        a, b = native.reset(seed=191101), computed.reset(seed=191101)
        np.testing.assert_array_equal(native.native.sim.data.qpos, computed.native.sim.data.qpos)
        assert native.obs_dim == computed.obs_dim
        assert native.metadata["xml_sha256"] == computed.metadata["xml_sha256"]
        np.testing.assert_array_equal(native.action_visibility, computed.action_visibility)
        changed = 0
        for step in range(40):
            # Zero controls establish contact; small varying controls then test
            # consecutive frames rather than a one-off force-buffer fill.
            command = np.full((native.n_agents, native.action_dim), .05*np.sin(step/4), np.float32)
            a, b = native.step(command), computed.step(command)
            original = native.native.sim.data
            fixed = computed.native.sim.data
            np.testing.assert_array_equal(original.qpos, fixed.qpos)
            np.testing.assert_array_equal(original.qvel, fixed.qvel)
            np.testing.assert_array_equal(a.terminated, b.terminated)
            np.testing.assert_array_equal(a.truncated, b.truncated)
            np.testing.assert_array_equal(a.observed_actions, b.observed_actions)
            np.testing.assert_array_equal(a.visibility, b.visibility)
            saved = {name: getattr(original, name).copy() for name in ("cfrc_ext", "cfrc_int", "cacc")}
            mujoco_py.functions.mj_rnePostConstraint(native.native.sim.model, original)
            np.testing.assert_array_equal(original.cfrc_ext, fixed.cfrc_ext)
            # Ant 8x1 has different hip/ankle widths at k=1; retain the
            # adapter's zero padding rather than assuming equal local lengths.
            expected_local = np.zeros((native.n_agents, native.raw_obs_dim), np.float32)
            for observer, local in enumerate(native._local_observations()):
                expected_local[observer, :len(local)] = local
            np.testing.assert_array_equal(b.obs[:, :computed.raw_obs_dim], expected_local)
            # Everything beyond the existing local raw features is unchanged.
            np.testing.assert_array_equal(a.obs[:, native.raw_obs_dim:], b.obs[:, computed.raw_obs_dim:])
            expected_contact = -.0005*np.square(np.clip(original.cfrc_ext, -1, 1)).sum()
            assert b.info["reward_contact"] == pytest.approx(expected_contact, abs=1e-14)
            expected_reward = (a.info["reward_forward"] + a.info["reward_ctrl"] +
                               a.info["reward_survive"] + expected_contact)
            assert b.rewards[0] == pytest.approx(expected_reward, abs=2e-7)
            assert a.info["reward_contact"] == 0
            changed += bool(np.any(a.obs != b.obs))
            for name, value in saved.items():
                getattr(original, name)[:] = value
        assert changed > 0
        metadata = computed.metadata["contact_force_computation"]
        assert computed.metadata["ant_contact_forces"] == "computed"
        assert computed.metadata["benchmark_track"] == "historical_gym_v2_ant_computed_forces_adaptation"
        assert metadata["after_control_substeps_before_reward"]
        assert metadata["after_reset_before_observation"]
        assert len(metadata["variant_source_sha256"]) == 64
        assert computed.metadata["historical_binary_parity_verified"] is False
    finally:
        native.close(); computed.close()


@native_only
def test_computed_reset_populates_contact_forces_before_exposing_local_observations():
    mujoco_py = pytest.importorskip("mujoco_py")
    env = LegacyMaMuJoCoEnv("Ant", "4x2", ant_contact_forces="computed")
    try:
        # The standard reset starts airborne, where both a computed force buffer
        # and a missing computation are zero. Lower this fixture's initial torso
        # so reset_model must populate forces at a contact-bearing initial state.
        env.native.init_qpos[2] = .3
        state = env.reset(seed=191101)
        assert env.native.sim.data.ncon > 0
        forces = env.native.sim.data.cfrc_ext.copy()
        assert forces.any()
        mujoco_py.functions.mj_rnePostConstraint(env.native.sim.model, env.native.sim.data)
        np.testing.assert_array_equal(env.native.sim.data.cfrc_ext, forces)
        np.testing.assert_array_equal(state.obs[:, :env.raw_obs_dim], np.stack(env._local_observations()))
        assert not state.visibility.any()
    finally:
        env.close()


@native_only
def test_computed_reset_recomputes_forces_and_keeps_seeded_reset_and_timeout():
    mujoco_py = pytest.importorskip("mujoco_py")
    env = LegacyMaMuJoCoEnv("Ant", "4x2", max_episode_steps=8, ant_contact_forces="computed")
    try:
        initial = env.reset(seed=191101)
        for _ in range(8):
            final = env.step(np.zeros((4, 2), np.float32))
        assert final.truncated.all() and not final.terminated.any()
        assert env.native.sim.data.cfrc_ext.any()
        reset = env.reset(seed=191101)
        np.testing.assert_array_equal(reset.obs, initial.obs)
        before = env.native.sim.data.cfrc_ext.copy()
        mujoco_py.functions.mj_rnePostConstraint(env.native.sim.model, env.native.sim.data)
        np.testing.assert_array_equal(env.native.sim.data.cfrc_ext, before)
        np.testing.assert_array_equal(reset.obs[:, :env.raw_obs_dim], np.stack(env._local_observations()))
        assert not reset.observed_actions.any() and not reset.visibility.any()
    finally:
        env.close()
