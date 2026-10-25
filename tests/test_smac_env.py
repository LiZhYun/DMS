"""SMAC adapter tests use an in-memory backend; no game is launched."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dms.smac_env import SMACEnv


class FakeSMAC:
    n_agents = 3
    n_actions = 4
    episode_limit = 2

    def __init__(self, **options):
        self.options = options
        self.close_calls = 0
        self.reset_calls = 0
        self.step_calls = 0
        self.alive = np.ones(3, dtype=bool)
        self.visible = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=bool)
        self.previous = np.zeros((3, 4), dtype=np.float32)
        self.next_alive = None
        self.next_available = None
        self.next_info = {"battle_won": False}
        self.next_done = False
        self.local_action_override = None
        self.available = np.array([[0, 1, 1, 1]] * 3, dtype=bool)

    def get_obs_size(self):
        return 2 + 1 * 2 + 2 * 5 + 1

    def get_obs_move_feats_size(self):
        return 2

    def get_obs_enemy_feats_size(self):
        return 1, 2

    def get_obs_ally_feats_size(self):
        return 2, 5

    def reset(self):
        self.reset_calls += 1
        self.previous.fill(0)
        return self.get_obs(), "privileged state must never enter EnvStep"

    def step(self, actions):
        self.step_calls += 1
        self.previous = np.eye(4, dtype=np.float32)[actions]
        if self.next_alive is not None:
            self.alive = self.next_alive.copy()
        if self.next_available is not None:
            self.available = self.next_available.copy()
        return 2.5, self.next_done, dict(self.next_info)

    def get_obs(self):
        observations = np.zeros((3, self.get_obs_size()), dtype=np.float32)
        for observer in range(3):
            if not self.alive[observer]:
                continue
            observations[observer, -1] = self.step_calls + 1
            for row, peer in enumerate(p for p in range(3) if p != observer):
                if self.visible[observer, peer] and self.alive[peer]:
                    start = 4 + row * 5
                    observations[observer, start] = 1
                    action = self.previous[peer]
                    if self.local_action_override is not None:
                        action = self.local_action_override[observer, peer]
                    observations[observer, start + 1 : start + 5] = action
        return observations

    def get_avail_actions(self):
        available = self.available.copy()
        available[~self.alive] = [1, 0, 0, 0]
        return available

    @property
    def last_action(self):
        raise AssertionError("Adapter must not broadcast the global last_action buffer")

    def get_state(self):
        raise AssertionError("Adapter must not request global state")

    def get_visibility_matrix(self):
        raise AssertionError("Use each native local ally visibility feature")

    def close(self):
        self.close_calls += 1


@pytest.fixture
def setup_env():
    backends = []

    def factory(**options):
        backend = FakeSMAC(**options)
        backends.append(backend)
        return backend

    env = SMACEnv("MMM2", seed=11, env_factory=factory)
    yield env, backends
    env.close()


def test_native_options_dimensions_and_reset_without_action_history(setup_env):
    env, backends = setup_env
    backend = backends[0]
    assert backend.reset_calls == 0
    assert backend.options["obs_last_action"] is True
    assert backend.options["continuing_episode"] is True
    assert backend.options["game_version"] == "4.10"
    assert env.n_agents == 3 and env.obs_dim == 23 and env.action_dim == 4
    assert env.action_dims == [4, 4, 4]
    assert env.discrete is True
    assert env.action_visibility.shape == (3, 3, 4)
    assert env.action_visibility.all()
    initial = env.reset()
    assert initial.obs.shape == (3, 23)
    assert initial.obs.dtype == np.float32
    assert initial.action_mask.dtype == bool
    assert initial.visibility.dtype == bool
    assert not initial.visibility.any()
    assert not initial.observed_actions.any()
    assert not initial.terminated.any() and not initial.truncated.any()
    assert not initial.rewards.any()
    assert initial.info["action_observability"] == "native_local_obs_last_action"


def test_peer_actions_use_local_observation_and_actual_visibility(setup_env):
    env, backends = setup_env
    backend = backends[0]
    env.reset()
    # Deliberately distinguish local action fields from the submitted actions.
    backend.local_action_override = np.broadcast_to(
        np.array([0, 1, 0, 0], dtype=np.float32), (3, 3, 4)
    )
    result = env.step(np.array([2, 3, 2]))
    expected_visibility = backend.visible | np.eye(3, dtype=bool)
    np.testing.assert_array_equal(result.visibility, expected_visibility)
    np.testing.assert_array_equal(result.observed_actions[0, 1], [0, 1, 0, 0])
    np.testing.assert_array_equal(result.observed_actions[1, 2], [0, 1, 0, 0])
    np.testing.assert_array_equal(result.observed_actions[0, 0], [0, 0, 1, 0])
    np.testing.assert_array_equal(result.obs[0, 15:19], [0, 0, 1, 0])
    assert not result.observed_actions[~expected_visibility].any()
    np.testing.assert_array_equal(result.rewards, [2.5, 2.5, 2.5])
    # The asymmetric local sensing pattern must not be symmetrized.
    assert result.visibility[0, 1] and not result.visibility[1, 0]


def test_current_legal_action_mask_does_not_erase_observed_previous_action(setup_env):
    env, backends = setup_env
    env.reset()
    backends[0].next_available = np.array([[0, 1, 0, 0]] * 3, dtype=bool)
    result = env.step(np.array([3, 3, 3]))
    assert not result.action_mask[:, 3].any()
    assert result.observed_actions[0, 0, 3] == 1
    assert result.observed_actions[0, 1, 3] == 1
    assert env.action_visibility.all()


def test_individual_death_does_not_terminate_shared_episode_or_recurrent_history(setup_env):
    env, backends = setup_env
    env.reset()
    backends[0].next_alive = np.array([True, False, True])
    result = env.step(np.array([1, 2, 3]))
    assert not result.terminated.any() and not result.truncated.any()
    assert not result.episode_done
    assert not result.visibility[1].any() and not result.visibility[:, 1].any()
    assert not result.observed_actions[1].any()
    np.testing.assert_array_equal(result.action_mask[1], [1, 0, 0, 0])
    env.step(np.array([1, 0, 1]))


@pytest.mark.parametrize(
    "info,terminated,truncated",
    [
        ({"battle_won": True}, True, False),
        ({"battle_won": False}, True, False),
        ({"battle_won": False, "episode_limit": True}, False, True),
    ],
)
def test_battle_end_and_timeout_preserve_terminal_observation(
    setup_env, info, terminated, truncated
):
    env, backends = setup_env
    env.reset()
    backend = backends[0]
    backend.next_done = True
    backend.next_info = info
    result = env.step(np.array([1, 2, 3]))
    assert np.all(result.terminated == terminated)
    assert np.all(result.truncated == truncated)
    assert result.episode_done
    np.testing.assert_array_equal(result.obs[:, 14], [2, 2, 2])
    assert result.info["battle_won"] == info["battle_won"]
    assert backend.reset_calls == 1  # No automatic reset overwrites final state.
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.array([1, 1, 1]))


def test_protocol_restart_is_not_recorded_as_a_game_loss(setup_env):
    env, backends = setup_env
    env.reset()
    backends[0].next_done = True
    backends[0].next_info = {}
    with pytest.raises(RuntimeError, match="infrastructure failure"):
        env.step(np.array([1, 2, 3]))


def test_explicit_reseed_recreates_backend_but_normal_reset_reuses_it(setup_env):
    env, backends = setup_env
    env.reset(seed=11)
    assert len(backends) == 1
    env.reset(seed=37)
    assert len(backends) == 2 and backends[0].close_calls == 1
    assert backends[1].options["seed"] == 37
    env.reset()
    assert len(backends) == 2
    env.reset(seed=37)
    assert len(backends) == 3 and backends[1].close_calls == 1
    env.close()
    env.close()
    assert backends[2].close_calls == 1


def test_process_lifetime_restart_waits_for_reset_and_preserves_final_state(setup_env):
    env, backends = setup_env
    env.reset(seed=11)
    backend = backends[0]
    backend._obs = SimpleNamespace(observation=SimpleNamespace(game_loop=500_000))
    backend.next_done = True
    backend.next_info = {'battle_won': True}
    final = env.step(np.array([1, 2, 3]))
    assert final.episode_done and final.info['battle_won']
    assert len(backends) == 1 and backend.close_calls == 0
    initial = env.reset()
    assert len(backends) == 2 and backend.close_calls == 1
    assert initial.info['scheduled_process_restart'] is True
    assert env.runtime_info['scheduled_restarts'] == 1
    assert env.runtime_info['seed_generation'] == 1
    assert backends[1].options['seed'] != 11
    assert not initial.visibility.any() and not initial.rewards.any()
    # The already returned final observation/reward is never replaced.
    assert final.info['battle_won'] and np.all(final.rewards == 2.5)


def test_scheduled_restart_seed_is_deterministic_and_explicit_reseed_restarts_schedule():
    def sequence():
        env = SMACEnv('MMM2', seed=11, env_factory=FakeSMAC)
        try:
            env.reset(seed=11)
            seeds = []
            for _ in range(2):
                env._env._obs = SimpleNamespace(observation=SimpleNamespace(game_loop=500_000))
                seeds.append(env.reset().info['process_seed'])
            env.reset(seed=11)
            assert env.runtime_info['seed_generation'] == 0
            env._env._obs = SimpleNamespace(observation=SimpleNamespace(game_loop=500_000))
            assert env.reset().info['process_seed'] == seeds[0]
            return seeds
        finally:
            env.close()
    first = sequence()
    assert first == sequence() and first[0] != first[1]


def test_normal_episode_reset_below_lifetime_threshold_reuses_process(setup_env):
    env, backends = setup_env
    env.reset()
    backends[0]._obs = SimpleNamespace(observation=SimpleNamespace(game_loop=499_999))
    assert not env.reset().info['scheduled_process_restart']
    assert len(backends) == 1 and env.runtime_info['scheduled_restarts'] == 0


@pytest.mark.parametrize('limit', [0, 524_000, 523_999])
def test_lifetime_threshold_leaves_room_for_a_complete_episode(limit):
    with pytest.raises(ValueError, match='max_game_loops'):
        SMACEnv('MMM2', env_factory=FakeSMAC, max_game_loops=limit)


@pytest.mark.parametrize(
    "actions",
    [np.array([1, 2]), np.array([1.1, 2, 3]), np.array([1, 2, 4]), np.array([0, 1, 1])],
)
def test_invalid_or_unavailable_actions_fail_before_backend_step(setup_env, actions):
    env, backends = setup_env
    env.reset()
    with pytest.raises(ValueError):
        env.step(actions)
    assert backends[0].step_calls == 0


@pytest.mark.parametrize(
    "option", ["obs_last_action", "continuing_episode"]
)
def test_required_native_settings_cannot_be_disabled(option):
    with pytest.raises(ValueError, match=option):
        SMACEnv("MMM2", env_factory=FakeSMAC, **{option: False})


def test_importing_module_does_not_import_smac_or_start_starcraft(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "smac" or name.startswith("smac."):
            raise AssertionError("SMAC import must wait for construction")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    import dms.smac_env

    importlib.reload(dms.smac_env)


def test_map_provenance_hashes_resolved_pysc2_asset_without_starting_game(tmp_path, monkeypatch):
    from dms.smac_env import _native_provenance

    relative = "SMAC_Maps/3s5z.SC2Map"
    path = tmp_path / "Maps" / relative
    path.parent.mkdir(parents=True)
    content = b"small map fixture\x00\xff"
    path.write_bytes(content)
    calls = []

    def select_version(*, version):
        calls.append(version)
        # No start() method exists: the provenance code must never launch SC2.
        return SimpleNamespace(data_dir=str(tmp_path), version=SimpleNamespace(
            game_version="4.10.0", build_version=75689, data_version="test-data-version",
        ))

    fake_pysc2 = SimpleNamespace(
        maps=SimpleNamespace(get=lambda name: SimpleNamespace(path=relative)),
        run_configs=SimpleNamespace(get=select_version),
    )
    monkeypatch.setitem(sys.modules, "pysc2", fake_pysc2)
    result = _native_provenance("3s5z", "4.10")
    assert calls == ["4.10"]
    assert result["map_asset"] == {
        "status": "resolved", "path": str(path.resolve()), "relative_path": relative,
        "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
    }
    assert result["runtime_selection"]["build_version"] == 75689
    assert result["runtime_selection"]["source"] == "pysc2_run_config_selection_not_launched_binary"

    path.unlink()
    missing = _native_provenance("3s5z", "4.10")
    assert missing["map_asset"]["status"] == "unresolved"
    assert "sha256" not in missing["map_asset"]


def test_unavailable_runtime_provenance_does_not_change_construction_contract(monkeypatch):
    from dms.smac_env import _native_provenance

    def unavailable(**kwargs):
        raise ValueError("unavailable version")

    monkeypatch.setitem(sys.modules, "pysc2", SimpleNamespace(
        maps=SimpleNamespace(), run_configs=SimpleNamespace(get=unavailable),
    ))
    result = _native_provenance("3s5z", "4.10")
    assert result["map_asset"]["status"] == "unresolved"
    assert "unavailable version" in result["map_asset"]["reason"]


def test_effective_default_metadata_uses_backend_values_without_inventing_mock_defaults():
    class EffectiveOptions(FakeSMAC):
        _step_mul = 8
        _move_amount = 2
        reward_sparse = False
        reward_scale = False
        reward_scale_rate = 20
        obs_own_health = True

    env = SMACEnv("MMM2", env_factory=EffectiveOptions, reward_scale=False, obs_own_health=False)
    try:
        requested = env.metadata["environment_options"]
        resolved = env.metadata["resolved_environment_options"]
        assert "step_mul" not in requested and resolved["step_mul"] == 8
        assert resolved["move_amount"] == 2 and resolved["episode_limit"] == 2
        assert resolved["reward_sparse"] is False and resolved["reward_scale"] is False
        assert requested["obs_own_health"] is False and resolved["obs_own_health"] is True
        assert "reward_win" not in resolved  # This mock does not expose that default.
        assert env.metadata["map_asset"]["status"] == "not_collected"
    finally:
        env.close()
