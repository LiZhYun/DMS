"""Historical Gym v2 physics with the paper's local multi-agent graph.

The supported stack is Gym 0.23.1, mujoco-py 2.1.2.14, and MuJoCo 2.1.
It is a historical compatibility track, not verified MuJoCo 2.0 parity.
Importing this module does not import Gym or compile a native extension.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from . import legacy_graph
from .envs import EnvStep, _augment_local_observations, _joint_action_visibility


_REWARDS = {
    "Ant": {
        "formula": "torso_x_velocity + 1 - 0.5*sum(control**2) - 0.0005*sum(clip(contact,-1,1)**2)",
        "alive_bonus_on_terminal": True,
        "healthy_height": [0.2, 1.0],
    },
    "HalfCheetah": {
        "formula": "root_x_velocity - 0.1*sum(control**2)",
        "native_termination": "never",
    },
    "Walker2d": {
        "formula": "root_x_velocity + 1 - 0.001*sum(control**2)",
        "alive_bonus_on_terminal": True,
        "healthy_height_open_interval": [0.8, 2.0],
        "healthy_angle_open_interval": [-1.0, 1.0],
    },
}
_ASSETS = {"Ant": "ant.xml", "HalfCheetah": "half_cheetah.xml", "Walker2d": "walker2d.xml"}


class LegacyMaMuJoCoEnv:
    """Original graph observations, corrected joint controls, local action history.

    Each component is routed through its graph joint's actuator index. Actors
    receive their graph observation, padding mask, public agent identity, and
    only previous executed controls on visible joints. Both IPPO and DMS receive
    those same channels. The complete simulator state is never an actor input.
    """

    discrete = False

    def __init__(
        self,
        scenario: str = "HalfCheetah",
        agent_conf: str = "2x3",
        agent_obsk: int = 1,
        max_episode_steps: int = 1000,
    ):
        if not isinstance(agent_obsk, int) or isinstance(agent_obsk, bool) or agent_obsk < 0:
            raise ValueError("agent_obsk must be a nonnegative integer")
        if not isinstance(max_episode_steps, int) or isinstance(max_episode_steps, bool) or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")
        scenario = scenario.removesuffix("-v2")
        self.agent_action_partitions, edges = legacy_graph.graph_for(scenario, agent_conf)
        self.k_dicts = [legacy_graph.distance_layers(part, edges, agent_obsk) for part in self.agent_action_partitions]
        self.k_categories = legacy_graph.observation_categories(scenario, agent_obsk)
        self.n_agents = len(self.agent_action_partitions)
        self.agents = [f"agent_{index}" for index in range(self.n_agents)]
        self.action_dims = [len(partition) for partition in self.agent_action_partitions]
        self.action_dim = max(self.action_dims)
        self.action_ids = [[joint.act_ids for joint in partition] for partition in self.agent_action_partitions]

        # Native imports are deliberately inside construction. Use the isolated
        # runtime helper; first mujoco-py import can compile an extension.
        for name, required in (("gym", "0.23.1"), ("mujoco-py", "2.1.2.14")):
            actual = importlib.metadata.version(name)
            if actual != required:
                raise RuntimeError(f"legacy backend requires {name}=={required}, found {actual}; use scripts/legacy/bootstrap.sh run")
        import gym
        import mujoco_py

        engine_version = int(mujoco_py.functions.mj_version())
        if engine_version != 210:
            raise RuntimeError(f"legacy compatibility backend requires MuJoCo 210, found {engine_version}")
        # Gym 0.23 forwards make() kwargs to the robot constructor, unlike
        # modern Gymnasium. Configure its existing TimeLimit wrapper directly.
        self.backend = gym.make(f"{scenario}-v2")
        self.backend._max_episode_steps = max_episode_steps
        self.backend.spec.max_episode_steps = max_episode_steps
        self.native = self.backend.unwrapped
        self._done = True
        self._elapsed_steps = 0
        try:
            all_ids = [index for partition in self.action_ids for index in partition]
            if sorted(all_ids) != list(range(self.native.action_space.shape[0])):
                raise ValueError("historical graph must assign each native actuator exactly once")
            self.action_low = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
            self.action_high = np.zeros_like(self.action_low)
            for agent, indices in enumerate(self.action_ids):
                self.action_low[agent, :len(indices)] = self.native.action_space.low[indices]
                self.action_high[agent, :len(indices)] = self.native.action_space.high[indices]
            self.action_visibility = _joint_action_visibility(self.agent_action_partitions, self.k_dicts, self.action_dim)
            self.observation_dims = [len(obs) for obs in self._local_observations()]
            self.raw_obs_dim = max(self.observation_dims)
            self._raw_valid = np.zeros((self.n_agents, self.raw_obs_dim), dtype=bool)
            for agent, size in enumerate(self.observation_dims):
                self._raw_valid[agent, :size] = True
            self.obs_dim = 2 * self.raw_obs_dim + self.n_agents + 2 * self.n_agents * self.action_dim

            asset = Path(inspect.getfile(type(self.native))).parent / "assets" / _ASSETS[scenario]
            graph_path = Path(legacy_graph.__file__)
            self.metadata = {
                "benchmark_track": "historical_gym_v2_mujoco210_compatibility",
                "historical_binary_parity_verified": False,
                "scenario": scenario,
                "native_env_id": f"{scenario}-v2",
                "agent_conf": agent_conf,
                "agent_obsk": agent_obsk,
                "k_categories": [list(categories) for categories in self.k_categories],
                "global_categories": [],
                "max_episode_steps": max_episode_steps,
                "timeout_decoding": "Gym TimeLimit.truncated; remaining done is terminal",
                "packages": {name: importlib.metadata.version(name) for name in ("gym", "mujoco-py", "numpy")},
                "mujoco_engine_version": engine_version,
                "xml_asset": asset.name,
                "xml_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                "original_graph_commit": legacy_graph.ORIGINAL_GRAPH_COMMIT,
                "original_graph_sha256": legacy_graph.ORIGINAL_GRAPH_SHA256,
                "adapted_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
                "action_routing": "graph_node_actuator_indices",
                "control_processing": "clip_native_bounds_before_simulation_and_reward",
                "action_ids": self.action_ids,
                "action_dims": self.action_dims,
                "observation_dims": self.observation_dims,
                "action_sensor": "previous_executed_control_on_observed_joints",
                "actor_information": "local_observation_public_id_visible_previous_controls_and_masks",
                "reward": _REWARDS[scenario].copy(),
                "frame_skip": self.native.frame_skip,
                "physics_timestep": float(self.native.model.opt.timestep),
                "native_geom_friction": self.native.model.geom_friction.tolist(),
            }
        except Exception:
            self.backend.close()
            raise

    def _local_observations(self) -> list[np.ndarray]:
        return [legacy_graph.local_observation(self.native.sim.data, layers, self.k_categories) for layers in self.k_dicts]

    def _transition(
        self, reward: float, terminated: bool, truncated: bool,
        info: dict[str, Any], executed: np.ndarray | None,
    ) -> EnvStep:
        raw = np.zeros((self.n_agents, self.raw_obs_dim), dtype=np.float32)
        for agent, observation in enumerate(self._local_observations()):
            raw[agent, :len(observation)] = observation
        components = self.action_visibility.copy()
        if executed is None:
            components.fill(False)
            observed = np.zeros_like(components, dtype=np.float32)
        else:
            observed = np.where(components, executed[None], 0.0).astype(np.float32)
        self._done = bool(terminated or truncated)
        return EnvStep(
            obs=_augment_local_observations(raw, self._raw_valid, observed, components),
            rewards=np.full(self.n_agents, reward, dtype=np.float32),
            terminated=np.full(self.n_agents, terminated, dtype=bool),
            truncated=np.full(self.n_agents, truncated, dtype=bool),
            action_mask=None,
            observed_actions=observed,
            visibility=components.any(axis=-1),
            info={**info, "elapsed_steps": self._elapsed_steps},
        )

    def reset(self, seed: int | None = None) -> EnvStep:
        self.backend.reset(seed=seed)
        self._elapsed_steps = 0
        return self._transition(0.0, False, False, {}, None)

    def step(self, actions: np.ndarray) -> EnvStep:
        if self._done:
            raise RuntimeError("reset the environment before stepping a new episode")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.n_agents, self.action_dim):
            raise ValueError(f"actions must have shape {(self.n_agents, self.action_dim)}")
        if not np.isfinite(actions).all():
            raise ValueError("actions must be finite")
        executed = np.clip(actions, self.action_low, self.action_high)
        native_action = np.zeros(self.native.action_space.shape, dtype=np.float32)
        for agent, indices in enumerate(self.action_ids):
            native_action[indices] = executed[agent, :len(indices)]
        _, reward, done, info = self.backend.step(native_action)
        self._elapsed_steps += 1
        truncated = bool(done and info.get("TimeLimit.truncated", False))
        return self._transition(float(reward), bool(done and not truncated), truncated, info, executed)

    def close(self) -> None:
        self.backend.close()
