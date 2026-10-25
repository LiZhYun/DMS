"""Bounded histories of the observations already available to each agent.

This wrapper changes only ``EnvStep.obs``. A history snapshot has shape
[agent, history, base observation + validity channel], with left zero padding.
``obs_dim`` continues to mean the base observation width used by the policy's
normalizer. The final channel is exactly 0/1 and is never normalized.

There is no learned hidden state here. Raw windows survive PPO updates; the
policy normalizes each complete window with its current rollout-frozen moments
and recomputes its GRU features during PPO. Every snapshot owns its memory.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np


class ObservationHistoryEnv:
    def __init__(self, env, length: int, *, clear_dead_agents: bool = False):
        if not isinstance(length, int) or isinstance(length, bool) or length < 1:
            raise ValueError('observation history length must be a positive integer')
        self.env, self.length = env, length
        self.clear_dead_agents = clear_dead_agents
        self._history = np.zeros((env.n_agents, length, env.obs_dim + 1), np.float32)

    def __getattr__(self, name):
        # Preserve native action routing, visibility, metadata, and runtime counters.
        return getattr(self.env, name)

    @property
    def metadata(self):
        metadata = deepcopy(getattr(self.env, 'metadata', {}))
        metadata['policy_observation_history'] = {
            'length': self.length,
            'base_observation_dim': self.env.obs_dim,
            'shape': [self.env.n_agents, self.length, self.env.obs_dim + 1],
            'validity_channel': 'last; exact 0/1; never normalized',
            'padding': 'left zeros; current observation at final slot',
            'reset': 'episode reset and own SMAC death' if self.clear_dead_agents else 'episode reset',
            'information': 'only this agent\'s existing local observations',
        }
        return metadata

    def _snapshot(self, state):
        raw = np.asarray(state.obs, dtype=np.float32)
        if raw.shape != (self.env.n_agents, self.env.obs_dim):
            raise ValueError('history wrapper requires base [agent, observation] inputs')
        active = np.ones(self.env.n_agents, dtype=bool)
        if self.clear_dead_agents:
            if state.action_mask is None or state.action_mask.shape != (self.env.n_agents, self.env.action_dim):
                raise ValueError('SMAC death reset requires the native available-action mask')
            # SMAC dead units have only no-op (action zero); live units can stop.
            active = state.action_mask[:, 1:].any(axis=-1)
        self._history[~active] = 0
        self._history[:, :-1] = self._history[:, 1:].copy()
        self._history[:, -1, :-1] = np.where(active[:, None], raw, 0)
        self._history[:, -1, -1] = active
        return replace(state, obs=self._history.copy())

    def reset(self, seed=None):
        state = self.env.reset(seed=seed)
        self._history.fill(0)
        return self._snapshot(state)

    def step(self, actions):
        # Preserve a timeout's final observation for bootstrap. Clearing occurs
        # only when the caller explicitly resets (or this observer is dead).
        return self._snapshot(self.env.step(actions))

    def close(self):
        return self.env.close()
