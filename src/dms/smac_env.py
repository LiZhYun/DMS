"""Official SMAC with intentions built only from native local action observations.

StarCraft is started by ``reset``, not by importing this module. Training and
evaluation should call ``reset()`` between episodes. An explicit ``reset(seed)``
recreates SMAC because its seed is fixed when the SC2 game is created.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np

from .envs import EnvStep


def _native_provenance(map_name: str, game_version: str) -> dict[str, Any]:
    """Resolve PySC2's actual map path and version selection without start().

    Map files are small; only their bytes are hashed, never the game binary.
    Missing installations remain constructible for dimension inspection, with
    an explicit unresolved record instead of invented provenance.
    """
    try:
        from pysc2 import maps, run_configs

        run_config = run_configs.get(version=game_version)
        descriptor = maps.get(map_name)
        path = Path(run_config.data_dir) / "Maps" / descriptor.path
        selection = {
            "source": "pysc2_run_config_selection_not_launched_binary",
            "game_version": run_config.version.game_version,
            "build_version": run_config.version.build_version,
            "data_version": run_config.version.data_version,
        }
        try:
            content = path.read_bytes()
            asset = {
                "status": "resolved",
                "path": str(path.resolve()),
                "relative_path": descriptor.path,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        except OSError as error:
            asset = {"status": "unresolved", "path": str(path), "reason": str(error)}
        return {"map_asset": asset, "runtime_selection": selection}
    except Exception as error:
        # Provenance must not alter SMAC's construction/launch behavior. Its
        # normal reset still reports missing maps, versions, or installations.
        return {"map_asset": {"status": "unresolved", "reason": f"{type(error).__name__}: {error}"}}


def _resolved_options(env: Any, requested: dict[str, Any]) -> dict[str, Any]:
    """Capture effective backend attributes, including unpassed defaults."""
    names = (
        "difficulty", "game_version", "continuing_episode", "obs_all_health", "obs_own_health",
        "obs_last_action", "obs_pathing_grid", "obs_terrain_height", "obs_instead_of_state",
        "obs_timestep_number", "state_last_action", "state_timestep_number", "reward_sparse",
        "reward_only_positive", "reward_death_value", "reward_win", "reward_defeat",
        "reward_negative_scale", "reward_scale", "reward_scale_rate", "heuristic_ai", "heuristic_rest",
    )
    attributes = {name: name for name in names}
    attributes.update(step_mul="_step_mul", move_amount="_move_amount", episode_limit="episode_limit")
    resolved = {key: value for key, value in requested.items() if key != "seed"}
    for name, attribute in attributes.items():
        if hasattr(env, attribute):
            resolved[name] = getattr(env, attribute)
    return resolved


class SMACEnv:
    """A small adapter for the original, discrete-action SMAC benchmark.

    ``env_factory`` is a test seam; ordinary use imports official SMAC lazily.
    Peer actions come from each observer's ally feature block with
    ``obs_last_action=True``. The complete runner action vector is used only
    for executing the joint step and each agent remembering its own action.
    """

    discrete = True

    def __init__(
        self,
        map_name: str,
        seed: int = 0,
        game_version: str = "4.10",
        difficulty: str = "7",
        *,
        max_game_loops: int = 500_000,
        env_factory: Callable[..., Any] | None = None,
        **smac_kwargs: Any,
    ) -> None:
        if not 0 < max_game_loops < 524_000:
            raise ValueError("max_game_loops must be between 1 and 523999")
        self.max_game_loops = int(max_game_loops)
        self._seed = int(seed)
        self._seed_generation = 0
        self._scheduled_restarts = 0
        required = {
            "obs_last_action": True,
            "continuing_episode": True,
            "obs_instead_of_state": False,
            "heuristic_ai": False,
        }
        for name, value in required.items():
            if name in smac_kwargs and smac_kwargs[name] != value:
                raise ValueError(f"DMS SMAC requires {name}={value!r}")
        official_backend = env_factory is None
        if official_backend:
            try:
                from smac.env import StarCraft2Env
            except ImportError as error:
                raise ImportError(
                    "Install the official SMAC package and the DMS smac extras "
                    "before constructing SMACEnv."
                ) from error
            env_factory = StarCraft2Env
        self._factory = env_factory
        self._options = dict(
            smac_kwargs,
            **required,
            map_name=map_name,
            game_version=game_version,
            difficulty=str(difficulty),
            seed=int(seed),
        )
        self._env = self._factory(**self._options)
        self.metadata = {
            'benchmark_track': 'original_smac',
            'map_name': map_name,
            'game_version': game_version,
            'difficulty': str(difficulty),
            'action_observability': 'native_local_obs_last_action',
            'environment_options': {key:value for key,value in self._options.items() if key != 'seed'},
            'resolved_environment_options': _resolved_options(self._env, self._options),
            'process_lifetime': {
                'restart_at_episode_boundary_after_game_loops': self.max_game_loops,
                'reason': 'SC2 cannot safely exceed 2**19 game loops in one game',
                'seed_schedule': 'SeedSequence([explicit_seed, generation]), uint32 modulo (2**31-1)',
            },
        }
        if official_backend:
            self.metadata.update(_native_provenance(map_name, game_version))
            self.metadata['packages'] = {}
            for package in ('smac', 'pysc2', 's2clientprotocol', 'protobuf'):
                try:
                    self.metadata['packages'][package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    pass
        else:
            self.metadata['map_asset'] = {'status': 'not_collected', 'reason': 'custom_env_factory'}
        self.n_agents = int(self._env.n_agents)
        self._raw_obs_dim = int(self._env.get_obs_size())
        self.action_dim = int(self._env.n_actions)
        # Native observations contain peer commands but omit the observer's
        # own previous command. Give every algorithm that same local memory,
        # a history-present bit, and public identity for shared policies.
        self.obs_dim = self._raw_obs_dim + self.action_dim + 1 + self.n_agents
        self.action_dims = [self.action_dim] * self.n_agents
        # All categorical coordinates exist. This is not dynamic peer visibility.
        self.action_visibility = np.ones(
            (self.n_agents, self.n_agents, self.action_dim), dtype=bool
        )
        self.episode_limit = int(self._env.episode_limit)
        step_mul = int(getattr(self._env, '_step_mul', self._options.get('step_mul', 8)))
        if self.max_game_loops + (self.episode_limit + 10) * step_mul >= 524_000:
            self._env.close()
            raise ValueError("max_game_loops must leave room for one complete episode before 524000")
        n_enemies, enemy_width = self._env.get_obs_enemy_feats_size()
        self._ally_start = (
            self._env.get_obs_move_feats_size() + n_enemies * enemy_width
        )
        n_allies, self._ally_width = self._env.get_obs_ally_feats_size()
        if n_allies != self.n_agents - 1 or self._ally_width < self.action_dim + 1:
            raise ValueError("Unexpected SMAC local ally observation layout")
        self._has_reset = False
        self._needs_reset = True
        self._closed = False

    @property
    def runtime_info(self) -> dict[str, Any]:
        observation = getattr(getattr(self._env, "_obs", None), "observation", None)
        game_loop = getattr(observation, "game_loop", None)
        return {
            'scheduled_restarts': self._scheduled_restarts,
            'seed_generation': self._seed_generation,
            'process_seed': self._options['seed'],
            'game_loop': int(game_loop) if game_loop is not None else None,
            'backend_force_restarts': int(getattr(self._env, 'force_restarts', 0)),
        }

    def reset(self, seed: int | None = None) -> EnvStep:
        if self._closed:
            raise RuntimeError("Cannot reset a closed SMAC environment")
        game_loop = self.runtime_info['game_loop']
        scheduled = seed is None and game_loop is not None and game_loop >= self.max_game_loops
        reseed = seed is not None and (
            self._has_reset or int(seed) != self._options["seed"]
        )
        if seed is not None:
            self._seed = int(seed)
            self._seed_generation = 0
        if scheduled or reseed:
            self._env.close()
            if scheduled:
                self._seed_generation += 1
                self._scheduled_restarts += 1
                seed = int(np.random.SeedSequence([self._seed, self._seed_generation])
                           .generate_state(1)[0]) % (2**31 - 1)
            self._options["seed"] = int(seed)
            self._env = self._factory(**self._options)
        self._env.reset()  # Native SMAC also returns state; it is intentionally unused.
        self._has_reset = True
        self._needs_reset = False
        return self._snapshot(
            reward=0.0,
            terminated=False,
            truncated=False,
            own_actions=None,
            info={
                "battle_won": False,
                "map_name": self._options["map_name"],
                "game_version": self._options["game_version"],
                "action_observability": "native_local_obs_last_action",
                "scheduled_process_restart": scheduled,
                "process_seed": self._options["seed"],
            },
        )

    def step(self, actions: np.ndarray) -> EnvStep:
        if self._needs_reset or self._closed:
            raise RuntimeError("Call reset before stepping a new SMAC episode")
        actions = np.asarray(actions)
        if actions.shape != (self.n_agents,) or not np.issubdtype(
            actions.dtype, np.integer
        ):
            raise ValueError("SMAC actions must be an integer array of shape [agents]")
        if np.any(actions < 0) or np.any(actions >= self.action_dim):
            raise ValueError("SMAC action index is outside the action space")
        available = np.asarray(self._env.get_avail_actions(), dtype=bool)
        if not np.all(available[np.arange(self.n_agents), actions]):
            raise ValueError("SMAC received an unavailable action")
        reward, done, native_info = self._env.step(actions.tolist())
        info = dict(native_info)
        if done and "battle_won" not in info:
            self._needs_reset = True
            raise RuntimeError(
                "SMAC ended without battle results, usually after an SC2 protocol "
                "failure. Do not count this infrastructure failure as a lost game."
            )
        truncated = bool(done and info.get("episode_limit", False))
        terminated = bool(done and not truncated)
        self._needs_reset = bool(done)
        return self._snapshot(reward, terminated, truncated, actions, info)

    def _snapshot(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
        own_actions: np.ndarray | None,
        info: dict[str, Any],
    ) -> EnvStep:
        obs = np.asarray(self._env.get_obs(), dtype=np.float32).copy()
        action_mask = np.asarray(self._env.get_avail_actions(), dtype=bool).copy()
        if obs.shape != (self.n_agents, self._raw_obs_dim):
            raise ValueError("SMAC returned an unexpected observation shape")
        if action_mask.shape != (self.n_agents, self.action_dim):
            raise ValueError("SMAC returned an unexpected available-action shape")
        observed_actions = np.zeros(
            (self.n_agents, self.n_agents, self.action_dim), dtype=np.float32
        )
        visibility = np.zeros((self.n_agents, self.n_agents), dtype=bool)
        if own_actions is not None:
            # Dead SMAC units have only action zero (no-op); live units can stop.
            alive = action_mask[:, 1:].any(axis=-1)
            end = self._ally_start + (self.n_agents - 1) * self._ally_width
            for observer in range(self.n_agents):
                peers = [peer for peer in range(self.n_agents) if peer != observer]
                allies = obs[observer, self._ally_start:end].reshape(
                    self.n_agents - 1, self._ally_width
                )
                visible_peers = (allies[:, 0] > 0.5) & alive[observer]
                visibility[observer, peers] = visible_peers
                observed_actions[observer, peers] = np.where(
                    visible_peers[:, None], allies[:, -self.action_dim :], 0.0
                )
                visibility[observer, observer] = alive[observer]
                if alive[observer]:
                    observed_actions[observer, observer, own_actions[observer]] = 1.0
        self_indices = np.arange(self.n_agents)
        obs = np.concatenate(
            (
                obs,
                observed_actions[self_indices, self_indices],
                visibility[self_indices, self_indices, None],
                np.eye(self.n_agents, dtype=np.float32),
            ),
            axis=-1,
        ).astype(np.float32, copy=False)
        # Flags describe the shared episode, not individual unit deaths. This
        # preserves recurrent histories until the caller explicitly resets.
        return EnvStep(
            obs=obs,
            rewards=np.full(self.n_agents, reward, dtype=np.float32),
            terminated=np.full(self.n_agents, terminated, dtype=bool),
            truncated=np.full(self.n_agents, truncated, dtype=bool),
            action_mask=action_mask,
            observed_actions=observed_actions,
            visibility=visibility,
            info=info,
        )

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True
            self._needs_reset = True
