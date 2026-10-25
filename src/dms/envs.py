"""Explicit local-information adapters and a small coordination smoke task.

The modern MaMuJoCo adapter adds a declared *local previous-control sensor*:
only actuator components whose joints belong to an observer's configured
observation graph are exposed. Joint position/velocity alone does not reveal
control commands; this is an information augmentation, not a claim about the
unmodified benchmark. Every algorithm receives the same sensor channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EnvStep:
    """Single-environment transition, with explicit agent/observer/peer axes.

    ``obs`` is [agent, observation], rewards and boundary flags are [agent].
    ``observed_actions`` is [observer, peer, action_feature], containing only
    controls observed before the next decision. ``visibility`` is
    [observer, peer]. Discrete action masks are [agent, action]. No adapter
    auto-resets, so final observations remain available for timeout bootstrap.
    """

    obs: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    action_mask: np.ndarray | None
    observed_actions: np.ndarray
    visibility: np.ndarray
    info: dict[str, Any]

    @property
    def episode_done(self) -> bool:
        return bool(np.all(self.terminated | self.truncated))


def _augment_local_observations(
    raw: np.ndarray,
    raw_valid: np.ndarray,
    observed_actions: np.ndarray,
    component_visibility: np.ndarray,
) -> np.ndarray:
    """Give shared-policy baselines the same local information as DMS."""
    agents = raw.shape[0]
    return np.concatenate(
        (
            raw,
            raw_valid,
            np.eye(agents, dtype=np.float32),
            observed_actions.reshape(agents, -1),
            component_visibility.reshape(agents, -1),
        ),
        axis=-1,
    ).astype(np.float32)


def _joint_action_visibility(
    partitions: list[Any], k_dicts: list[dict[int, list[Any]]], action_dim: int
) -> np.ndarray:
    """Mask individual actuators; seeing one peer joint exposes only that joint."""
    agents = len(partitions)
    mask = np.zeros((agents, agents, action_dim), dtype=bool)
    for observer, distances in enumerate(k_dicts):
        visible_joints = {joint for joints in distances.values() for joint in joints}
        for peer, partition in enumerate(partitions):
            for component, joint in enumerate(partition):
                mask[observer, peer, component] = joint in visible_joints
    return mask


def _paper_factorization(scenario: str, agent_conf: str) -> dict[str, Any] | None:
    """Restore paper partitions absent from Gymnasium-Robotics' public presets.

    Joint membership follows the original DMS graph, while current node objects
    preserve the modern adapter's explicit actuator-index routing. In particular,
    the middle 3x2 agent controls one joint from each leg; grouping two contiguous
    actuator indices would silently define a different partition.
    """
    if (scenario, agent_conf) not in {
        ("Ant", "8x1"), ("HalfCheetah", "3x2"), ("Walker2d", "3x2"), ("Walker2d", "6x1")
    }:
        return None
    from gymnasium_robotics.envs.multiagent_mujoco.obsk import get_parts_and_edges

    whole, edges, global_nodes = get_parts_and_edges(scenario, None)
    nodes = {node.label: node for partition in whole for node in partition}
    if scenario == "Ant":
        order = tuple(label for leg in range(1, 5) for label in (f"hip{leg}", f"ankle{leg}"))
    elif scenario == "HalfCheetah":
        order = ("bfoot", "bshin", "bthigh", "ffoot", "fshin", "fthigh")
    else:
        order = ("foot_joint", "leg_joint", "thigh_joint", "foot_left_joint", "leg_left_joint", "thigh_left_joint")
    size = int(agent_conf.split("x")[1])
    partitions = [tuple(nodes[label] for label in order[start : start + size]) for start in range(0, len(order), size)]
    return {"partition": partitions, "edges": edges, "globals": global_nodes}


class MaMuJoCoEnv:
    """Gymnasium-Robotics MaMuJoCo adaptation with explicit local action sensors.

    The installed Robotics implementation constructs Gymnasium MuJoCo v5 tasks.
    This is a modern adaptation, not verified parity with the paper's simulator.
    Action and observation spaces can differ across agents: both are padded,
    and their true sizes/masks remain explicit. Policies store their original
    Gaussian sample for likelihood evaluation; only commands sent to the
    environment are clipped. Sensors report those executed commands.

    ``backend`` accepts a compatible parallel environment for adapter testing
    or a custom factorization. Normal use leaves it unset.
    """

    discrete = False

    def __init__(
        self,
        scenario: str = "HalfCheetah",
        agent_conf: str = "2x3",
        agent_obsk: int = 0,
        max_episode_steps: int = 1000,
        *,
        global_categories: tuple[str, ...] = (),
        backend: Any | None = None,
        **env_kwargs: Any,
    ):
        if agent_obsk < 0:
            raise ValueError("agent_obsk must be nonnegative")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if backend is None:
            from gymnasium_robotics.envs.multiagent_mujoco.mujoco_multi import MultiAgentMujocoEnv

            if "agent_factorization" not in env_kwargs:
                factorization = _paper_factorization(scenario, agent_conf)
                if factorization is not None:
                    env_kwargs["agent_factorization"] = factorization
            backend = MultiAgentMujocoEnv(
                scenario=scenario,
                agent_conf=agent_conf,
                agent_obsk=agent_obsk,
                global_categories=global_categories,
                max_episode_steps=max_episode_steps,
                **env_kwargs,
            )
        self.backend = backend
        self.agents = list(backend.possible_agents)
        self.n_agents = len(self.agents)
        self.action_dims = [int(backend.action_space(agent).shape[0]) for agent in self.agents]
        self.action_dim = max(self.action_dims)
        self.observation_dims = [int(backend.observation_space(agent).shape[0]) for agent in self.agents]
        self.raw_obs_dim = max(self.observation_dims)
        self._raw_valid = np.zeros((self.n_agents, self.raw_obs_dim), dtype=bool)
        self.action_low = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
        self.action_high = np.zeros_like(self.action_low)
        for index, agent in enumerate(self.agents):
            self._raw_valid[index, : self.observation_dims[index]] = True
            space = backend.action_space(agent)
            self.action_low[index, : self.action_dims[index]] = space.low
            self.action_high[index, : self.action_dims[index]] = space.high
        if hasattr(backend, "k_dicts"):
            self.action_visibility = _joint_action_visibility(
                backend.agent_action_partitions, backend.k_dicts, self.action_dim
            )
        elif self.n_agents == 1:
            self.action_visibility = np.ones((1, 1, self.action_dim), dtype=bool)
        else:
            raise ValueError("multi-agent backend must expose its local joint visibility graph")
        self.obs_dim = 2 * self.raw_obs_dim + self.n_agents + 2 * self.n_agents * self.action_dim
        self.metadata = {
            "benchmark_track": "modern_mamujoco_adaptation",
            "scenario": scenario,
            "agent_conf": agent_conf,
            "agent_obsk": agent_obsk,
            "global_categories": list(global_categories),
            "max_episode_steps": max_episode_steps,
            "action_sensor": "previous_executed_control_on_observed_joints",
            "action_dims": self.action_dims,
            "observation_dims": self.observation_dims,
            "actor_information": "local_observation_public_id_visible_previous_controls_and_masks",
        }
        self._done = True

    def _transition(
        self,
        observations: dict[str, np.ndarray],
        rewards: dict[str, float],
        terminated: dict[str, bool],
        truncated: dict[str, bool],
        info: dict[str, Any],
        executed_actions: np.ndarray | None,
    ) -> EnvStep:
        raw = np.zeros((self.n_agents, self.raw_obs_dim), dtype=np.float32)
        for index, agent in enumerate(self.agents):
            raw[index, : self.observation_dims[index]] = observations[agent]
        components = self.action_visibility.copy()
        if executed_actions is None:
            components.fill(False)
            observed = np.zeros((self.n_agents, self.n_agents, self.action_dim), dtype=np.float32)
        else:
            observed = np.where(components, executed_actions[None], 0.0).astype(np.float32)
        transition = EnvStep(
            obs=_augment_local_observations(raw, self._raw_valid, observed, components),
            rewards=np.array([rewards[agent] for agent in self.agents], dtype=np.float32),
            terminated=np.array([terminated[agent] for agent in self.agents], dtype=bool),
            truncated=np.array([truncated[agent] for agent in self.agents], dtype=bool),
            action_mask=None,
            observed_actions=observed,
            visibility=components.any(axis=-1),
            info=info,
        )
        self._done = transition.episode_done
        return transition

    def reset(self, seed: int | None = None) -> EnvStep:
        observations, info = self.backend.reset(seed=seed)
        return self._transition(
            observations,
            dict.fromkeys(self.agents, 0.0),
            dict.fromkeys(self.agents, False),
            dict.fromkeys(self.agents, False),
            info,
            None,
        )

    def step(self, actions: np.ndarray) -> EnvStep:
        if self._done:
            raise RuntimeError("reset the environment before stepping a new episode")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.n_agents, self.action_dim):
            raise ValueError(f"actions must have shape {(self.n_agents, self.action_dim)}")
        if not np.isfinite(actions).all():
            raise ValueError("actions must be finite")
        executed = np.clip(actions, self.action_low, self.action_high)
        commands = {agent: executed[index, : self.action_dims[index]] for index, agent in enumerate(self.agents)}
        observations, rewards, terminated, truncated, info = self.backend.step(commands)
        return self._transition(observations, rewards, terminated, truncated, info, executed)

    def close(self) -> None:
        self.backend.close()


class CoordinationToy:
    """Tiny delayed-signal task for integration checks, never a paper benchmark.

    A leader observes an episode's binary target. Other agents receive their
    predecessor's previous action around a directed ring. Every agent receives
    the mean fraction of correct choices. This requires learning to propagate
    behavior across time, while keeping information access easy to inspect.
    """

    discrete = True
    action_dim = 2

    def __init__(self, n_agents: int = 3, horizon: int = 32):
        if n_agents < 2 or horizon < 1:
            raise ValueError("toy requires at least two agents and a positive horizon")
        self.n_agents = n_agents
        self.horizon = horizon
        self.action_dims = [self.action_dim] * n_agents
        self.action_visibility = np.zeros((n_agents, n_agents, self.action_dim), dtype=bool)
        for observer in range(n_agents):
            self.action_visibility[observer, observer] = True
            self.action_visibility[observer, (observer - 1) % n_agents] = True
        self.obs_dim = 6 + n_agents + 2 * n_agents * self.action_dim
        self.metadata = {
            "benchmark_track": "synthetic_integration_only",
            "task": "leader_signal_propagation",
            "n_agents": n_agents,
            "horizon": horizon,
            "action_sensor": "own_and_ring_predecessor_previous_action",
        }
        self._rng = np.random.default_rng()
        self._done = True

    def _transition(self, actions: np.ndarray | None) -> EnvStep:
        raw = np.zeros((self.n_agents, 3), dtype=np.float32)
        raw[0, self._target] = 1.0
        raw[:, 2] = self._time / self.horizon
        components = self.action_visibility.copy()
        if actions is None:
            components.fill(False)
            observed = np.zeros((self.n_agents, self.n_agents, self.action_dim), dtype=np.float32)
            reward = 0.0
            success = False
        else:
            one_hot = np.eye(self.action_dim, dtype=np.float32)[actions]
            observed = np.where(components, one_hot[None], 0.0).astype(np.float32)
            reward = float(np.mean(actions == self._target))
            success = bool(np.all(actions == self._target))
        self._done = self._time >= self.horizon
        return EnvStep(
            obs=_augment_local_observations(raw, np.ones_like(raw, dtype=bool), observed, components),
            rewards=np.full(self.n_agents, reward, dtype=np.float32),
            terminated=np.zeros(self.n_agents, dtype=bool),
            truncated=np.full(self.n_agents, self._done, dtype=bool),
            action_mask=np.ones((self.n_agents, self.action_dim), dtype=bool),
            observed_actions=observed,
            visibility=components.any(axis=-1),
            info={"success": success},
        )

    def reset(self, seed: int | None = None) -> EnvStep:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._target = int(self._rng.integers(self.action_dim))
        self._time = 0
        return self._transition(None)

    def step(self, actions: np.ndarray) -> EnvStep:
        if self._done:
            raise RuntimeError("reset the environment before stepping a new episode")
        actions = np.asarray(actions)
        if actions.shape != (self.n_agents,):
            raise ValueError(f"actions must have shape {(self.n_agents,)}")
        if not np.isfinite(actions).all() or np.any(actions != np.rint(actions)):
            raise ValueError("discrete actions must be finite integer indices")
        if np.any(actions < 0) or np.any(actions >= self.action_dim):
            raise ValueError("discrete action index outside the action space")
        self._time += 1
        return self._transition(actions.astype(np.int64))

    def close(self) -> None:
        pass


def make_env(name: str, **kwargs: Any) -> CoordinationToy | MaMuJoCoEnv:
    """Construct one environment without vectorization or hidden auto-reset."""
    if name == "toy":
        return CoordinationToy(**kwargs)
    if name == "mamujoco":
        return MaMuJoCoEnv(**kwargs)
    raise ValueError(f"unknown environment {name!r}; choose 'toy' or 'mamujoco'")
