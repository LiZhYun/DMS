"""One readable on-policy loop for IPPO and DMS.

Read in order: Config -> Learner -> collect -> update -> evaluate -> main.
Rollouts use [time, environment, observer, ...]. Intention histories add
[history, peer, action_feature]. Inference features/gates are cached for PPO;
the auxiliary model is updated after PPO. Optional joint training re-encodes
local past histories during PPO, keeping rollout gates and filtering fixed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
import importlib.metadata
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
import torch
from torch import nn

from .envs import make_env
from .diagnostics import forecast_diagnostics, summarize_forecasts
from .intention import IntentionModel
from .policy import IndependentAgent
from .parallel import step_environments
from .returns import generalized_advantage_estimate


@dataclass
class Config:
    algorithm: str = 'dms'
    env: str = 'toy'
    mamujoco_backend: str = 'modern'
    scenario: str = 'HalfCheetah'
    ant_contact_forces: str = 'native'
    agent_conf: str = '2x3'
    agent_obsk: int = 1
    map_name: str = '3s5z'
    seed: int = 101
    device: str = 'cpu'
    num_envs: int = 4
    parallel_env_steps: bool = False
    mask_inactive_agents: bool = False
    smac_time_limit_terminal: bool = False
    update_mode: str = 'independent'
    total_steps: int = 4096
    rollout_steps: int = 128
    epochs: int = 5
    minibatch_size: int = 256
    hidden_dim: int = 64
    policy_history_length: int = 0
    initial_log_std: float = -0.5
    latent_dim: int = 8
    decoder_history: bool = True
    history_length: int = 32
    prediction_horizon: int = 4
    coordination: str = 'soft'
    coordination_threshold: float = 0.6
    short_history: int = 4
    filter_history: bool = True
    intention_threshold: float = 1e4
    warmup_episodes: int = 0
    causal_warmup_episodes: int = 100
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    intention_lr: float = 1e-3
    intention_rl_lr: float = 0.0
    aux_updates: int = 2
    aux_batch_size: int = 64
    kl_beta: float = 1.0
    latent_samples: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    eval_every: int = 2048
    eval_episodes: int = 10
    eval_seed: int = 10001
    output: str = 'runs/toy-dms-101'
    torch_threads: int = 1

    def validate(self):
        if (isinstance(self.intention_rl_lr, bool)
                or not isinstance(self.intention_rl_lr, (int, float))
                or not math.isfinite(self.intention_rl_lr) or self.intention_rl_lr < 0):
            raise ValueError('intention_rl_lr must be finite and nonnegative')
        if self.intention_rl_lr and self.algorithm == 'dms' and self.coordination == 'causal':
            raise ValueError('intention_rl_lr currently requires soft or all coordination')
        if not isinstance(self.smac_time_limit_terminal, bool):
            raise ValueError('smac_time_limit_terminal must be boolean')
        if self.smac_time_limit_terminal and self.env != 'smac':
            raise ValueError('smac_time_limit_terminal requires env=smac')
        if self.update_mode not in ('independent', 'sequential'):
            raise ValueError('update_mode must be independent or sequential')
        if self.ant_contact_forces not in ('native', 'computed'):
            raise ValueError('ant_contact_forces must be native or computed')
        if self.ant_contact_forces == 'computed' and (
                self.env != 'mamujoco' or self.mamujoco_backend != 'legacy' or self.scenario != 'Ant'):
            raise ValueError('computed contact forces require historical legacy Ant')
        if (isinstance(self.initial_log_std, bool)
                or not isinstance(self.initial_log_std, (float, int))
                or not -5 <= self.initial_log_std <= 2):
            raise ValueError('initial_log_std must be finite and in [-5, 2]')
        if (not isinstance(self.policy_history_length, int)
                or isinstance(self.policy_history_length, bool) or self.policy_history_length < 0):
            raise ValueError('policy_history_length must be a nonnegative integer')
        if self.algorithm not in {'ippo', 'dms'}:
            raise ValueError('algorithm must be ippo or dms')
        if self.env not in {'toy', 'mamujoco', 'smac'}:
            raise ValueError('env must be toy, mamujoco, or smac')
        if self.mamujoco_backend not in {'modern', 'legacy'}:
            raise ValueError('mamujoco_backend must be modern or legacy')
        if self.coordination not in {'soft', 'causal', 'all'}:
            raise ValueError('coordination must be soft, causal, or all')
        for name in ['num_envs','total_steps','rollout_steps','epochs','minibatch_size',
                     'history_length','prediction_horizon','eval_every','eval_episodes',
                     'hidden_dim','latent_dim','latent_samples','short_history','torch_threads']:
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        for name in ['aux_updates','warmup_episodes','causal_warmup_episodes','intention_threshold']:
            if getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative')
        if self.aux_updates and self.aux_batch_size < 1:
            raise ValueError('aux_batch_size must be positive when auxiliary updates are enabled')
        for name in ['gamma','gae_lambda']:
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        for name in ['clip','value_clip','max_grad_norm']:
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.prediction_horizon > self.rollout_steps:
            raise ValueError('prediction_horizon cannot exceed rollout_steps')
        if self.total_steps % self.num_envs:
            raise ValueError('total_steps must be divisible by num_envs')


def create_env(config: Config):
    if config.env == 'smac':
        from .smac_env import SMACEnv
        env = SMACEnv(config.map_name, seed=config.seed)
    elif config.env == 'mamujoco':
        if config.mamujoco_backend == 'legacy':
            from .legacy_env import LegacyMaMuJoCoEnv
            env = LegacyMaMuJoCoEnv(config.scenario,config.agent_conf,config.agent_obsk,
                                   ant_contact_forces=config.ant_contact_forces)
        else:
            env = make_env('mamujoco', scenario=config.scenario, agent_conf=config.agent_conf,
                           agent_obsk=config.agent_obsk)
    elif config.env == 'toy':
        env = make_env('toy')
    else:
        raise ValueError(f'unknown environment: {config.env}')
    if config.policy_history_length:
        from .observation_history import ObservationHistoryEnv
        try:
            env = ObservationHistoryEnv(env, config.policy_history_length,
                                        clear_dead_agents=config.env == 'smac')
        except BaseException:
            env.close()
            raise
    return env


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def runtime_metadata(device: str) -> dict:
    """Record the hardware used; a seed alone does not ensure cross-GPU replay."""
    result = {
        'host': platform.node(), 'device': device,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'slurm_partition': os.environ.get('SLURM_JOB_PARTITION'),
        'torch_threads': torch.get_num_threads(),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'accelerator': None,
    }
    if torch.device(device).type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        result['accelerator'] = {
            'name': properties.name,
            'capability': [properties.major, properties.minor],
            'total_memory_bytes': properties.total_memory,
            'cuda_runtime': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        }
    return result


def append_json(path: Path, record: dict):
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')


class Learner:
    def __init__(self, config: Config, env):
        self.config, self.device = config, torch.device(config.device)
        self.num_agents, self.action_dim = env.n_agents, env.action_dim
        self.action_dims = list(env.action_dims)
        self.discrete = env.discrete
        self.completed_episodes = 0
        self.latent_dim = config.latent_dim if config.algorithm == 'dms' else 0
        self.agents = nn.ModuleList([
            IndependentAgent(env.obs_dim, env.action_dim if env.discrete else env.action_dims[i], env.n_agents, self.latent_dim,
                             config.hidden_dim, env.discrete, config.coordination, i,
                             policy_history_length=config.policy_history_length,
                             initial_log_std=config.initial_log_std)
            for i in range(env.n_agents)
        ]).to(self.device)
        self.intentions = nn.ModuleList([
            IntentionModel(env.n_agents, env.action_dim, config.hidden_dim, config.latent_dim,
                           env.discrete, env.action_dims,
                           observation_mask=None if env.discrete else env.action_visibility[i],
                           decoder_history=config.decoder_history)
            for i in range(env.n_agents)
        ]).to(self.device) if self.latent_dim else nn.ModuleList()
        self.optimizers = []
        for i, agent in enumerate(self.agents):
            critic_parameters = list(agent.critic_parameters())
            critic_ids = {id(p) for p in critic_parameters}
            actor_parameters = [p for p in agent.parameters() if id(p) not in critic_ids]
            groups = [
                {'params': actor_parameters, 'lr': config.actor_lr},
                {'params': critic_parameters, 'lr': config.critic_lr},
            ]
            if self.latent_dim and config.intention_rl_lr:
                # Alternating RL/ELBO updates deliberately use separate Adam moments.
                groups.append({'params': list(self.intentions[i].prior_parameters()),
                               'lr': config.intention_rl_lr})
            self.optimizers.append(torch.optim.Adam(groups, eps=1e-5))
        self.aux_optimizers = [torch.optim.Adam(m.parameters(), lr=config.intention_lr)
                               for m in self.intentions]

    def tensor(self, array, dtype=torch.float32):
        return torch.as_tensor(array, dtype=dtype, device=self.device)

    @torch.no_grad()
    def contexts(self, states, history, history_visible, use_intentions=True):
        raw_obs = self.tensor(np.stack([state.obs for state in states]))
        history, history_visible = self.tensor(history), self.tensor(history_visible, torch.bool)
        observations, contexts, gates, available = [], [], [], []
        for i, agent in enumerate(self.agents):
            obs = agent.normalize_obs(raw_obs[:, i])
            if self.latent_dim and use_intentions:
                past, visible = history[:, :, i], history_visible[:, :, i]
                if self.config.filter_history:
                    visible, _ = self.intentions[i].filtered_history(
                        past, visible, self.config.short_history, self.config.intention_threshold)
                context = self.intentions[i].features(past, visible)
                peer_visible = visible.any(dim=1)
            else:
                context = obs.new_zeros(len(states), self.num_agents, self.latent_dim)
                peer_visible = torch.zeros(len(states), self.num_agents, dtype=torch.bool, device=self.device)
            mask = self.tensor(np.stack([s.action_mask[i] for s in states]), torch.bool) if self.discrete else None
            gate = agent.relevance(obs, context, peer_visible, mask, self.config.coordination_threshold)
            if (self.latent_dim and self.config.coordination == 'causal' and use_intentions
                    and self.completed_episodes < self.config.warmup_episodes + self.config.causal_warmup_episodes):
                # Explicit redesign: learn sensitivity before hard selection.
                # Set causal_warmup_episodes=0 for the manuscript threshold from activation.
                gate = peer_visible.clone()
                gate[:,i] = False
            observations.append(obs); contexts.append(context); gates.append(gate)
            if mask is not None:
                available.append(mask)
        return (torch.stack(observations, 1), torch.stack(contexts, 1),
                torch.stack(gates, 1), torch.stack(available, 1) if available else None)

    @torch.no_grad()
    def action(self, observations, contexts, gates, available, deterministic=False):
        result = [agent.act(observations[:, i], contexts[:, i], gates[:, i],
                            available[:, i] if available is not None else None, deterministic)
                  for i, agent in enumerate(self.agents)]
        actions = [row[0] for row in result]
        if not self.discrete:
            actions = [torch.nn.functional.pad(action,(0,self.action_dim-action.shape[-1])) for action in actions]
        return (torch.stack(actions,1), torch.stack([row[1] for row in result],1),
                torch.stack([row[2] for row in result],1))

    def save(self, path: Path, steps: int, episodes: int):
        checkpoint = {
            'config': asdict(self.config), 'steps': steps, 'episodes': episodes,
            'agents': self.agents.state_dict(), 'intentions': self.intentions.state_dict(),
            'optimizers': [o.state_dict() for o in self.optimizers],
            'aux_optimizers': [o.state_dict() for o in self.aux_optimizers],
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'environment_state_saved': False,
            'note': 'Evaluation checkpoint; restarting environments is not bitwise trajectory continuation.',
        }
        temporary = path.with_suffix('.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(path)


def empty_history(learner: Learner, environments: int):
    c, a, d = learner.config, learner.num_agents, learner.action_dim
    return (np.zeros((environments, c.history_length, a, a, d), np.float32),
            np.zeros((environments, c.history_length, a, a), bool))


def advance_history(history, visible, next_states):
    history[:, :-1] = history[:, 1:].copy()
    visible[:, :-1] = visible[:, 1:].copy()
    history[:, -1] = np.stack([s.observed_actions for s in next_states])
    visible[:, -1] = np.stack([s.visibility for s in next_states])


def collect(learner, environments, states, history, visible, episodes, episode_returns, rollout_steps=None,
            executor=None):
    """Keep final observations for bootstrapping; reset only after evaluating them."""
    config = learner.config
    records = {name: [] for name in ['obs','raw_obs','context','gates','actions','log_prob','values',
               'rewards','next_values','terminated','truncated','history','visible','future','future_visible','ends']}
    if learner.discrete:
        records['available'] = []
    finished = []
    cached_inputs = None
    for _ in range(rollout_steps or config.rollout_steps):
        learner.completed_episodes = episodes
        inputs = cached_inputs if cached_inputs is not None else learner.contexts(
            states, history, visible, episodes >= config.warmup_episodes)
        obs, context, gates, available = inputs
        actions, log_prob, values = learner.action(*inputs)
        action_array = actions.cpu().numpy()
        next_states = step_environments(environments, action_array, executor)
        next_history, next_visible = history.copy(), visible.copy()
        advance_history(next_history, next_visible, next_states)
        next_inputs = learner.contexts(next_states, next_history, next_visible,
                                       episodes >= config.warmup_episodes)
        with torch.no_grad():
            next_values = torch.stack([agent.value(next_inputs[0][:, i], next_inputs[1][:, i], next_inputs[2][:, i])
                                       for i, agent in enumerate(learner.agents)], 1)
        step = dict(obs=obs, context=context, gates=gates, actions=actions,
                    log_prob=log_prob, values=values, next_values=next_values)
        for key, value in step.items():
            records[key].append(value.detach())
        for key, value in [('raw_obs',np.stack([s.obs for s in states])), ('history',history.copy()),
                           ('visible',visible.copy()), ('future',np.stack([s.observed_actions for s in next_states])),
                           ('future_visible',np.stack([s.visibility for s in next_states]))]:
            records[key].append(value)
        for key in ['rewards','terminated','truncated']:
            records[key].append(np.stack([getattr(s, key) for s in next_states]))
        records['ends'].append(np.array([s.episode_done for s in next_states]))
        # Models and observation statistics stay fixed throughout collection.
        # Bootstrap inputs are therefore also the next decision's inputs.
        # Any reset may change warmup phase, so recompute the whole batch then.
        # The cache is local to this rollout: PPO/auxiliary updates invalidate it.
        cached_inputs = None if records['ends'][-1].any() else next_inputs
        if available is not None:
            records['available'].append(available)
        history, visible = next_history, next_visible
        for e, state in enumerate(next_states):
            episode_returns[e] += float(state.rewards.mean())
            if state.episode_done:
                finished.append({'return': episode_returns[e], 'battle_won': state.info.get('battle_won')})
                episodes += 1
                episode_returns[e] = 0
                next_states[e] = environments[e].reset()
                history[e] = 0; visible[e] = False
        states = next_states
    for key in records:
        records[key] = torch.stack(records[key]) if isinstance(records[key][0], torch.Tensor) else np.stack(records[key])
    return records, states, history, visible, episodes, episode_returns, finished



@torch.no_grad()
def prepare_intention_inputs(learner, data, agent_index):
    """Freeze local history filtering before this observer's first PPO update.

    Return [time*environment, history, peer, (action)] tensors. Future targets
    never enter this path. Rollout gates remain fixed even across warmup changes.
    """
    c, i = learner.config, agent_index
    if not (learner.latent_dim and c.intention_rl_lr and data['gates'][:, :, i].any()):
        return None
    past = learner.tensor(data['history'][:, :, :, i]).flatten(0, 1)
    visible = learner.tensor(data['visible'][:, :, :, i], torch.bool).flatten(0, 1)
    if c.filter_history:
        visible = torch.cat([learner.intentions[i].filtered_history(
            past[rows], visible[rows], c.short_history, c.intention_threshold)[0]
            for rows in torch.arange(len(past), device=learner.device).split(c.minibatch_size)])
    return past, visible


def ppo_context(learner, agent_index, cached, gates, intention_inputs, indices):
    """Differentiate through the current prior, or reuse the frozen default."""
    if intention_inputs is None or not gates[indices].any():
        # Avoid zero-gradient Adam momentum updates when no peer is used.
        return cached[indices]
    past, visible = intention_inputs
    return learner.intentions[agent_index].features(past[indices], visible[indices])


def sequential_factor(log_correction):
    """Detached predecessor likelihood product, without clipping or rescaling."""
    factor = log_correction.detach().exp()
    if not torch.isfinite(log_correction).all() or not torch.isfinite(factor).all() or not (factor > 0).all():
        raise FloatingPointError('Nonfinite or underflowed sequential PPO correction')
    return factor


@torch.no_grad()
def advance_sequential_correction(learner, data, agent_index, log_correction, intention_inputs=None):
    """Reevaluate stored actions after one agent finishes all its PPO epochs.

    Each distribution already returns the joint action log probability. Keep
    the full time/environment row index even when this agent trained on only
    its active subset. Cached normalized observations, contexts and gates keep
    the likelihood comparison conditional on the same rollout information. Joint
    training re-encodes that history with updated parameters and the old filter.
    """
    i, agent = agent_index, learner.agents[agent_index]
    if (learner.latent_dim and learner.config.intention_rl_lr
            and data['gates'][:, :, i].any() and intention_inputs is None):
        raise ValueError('Joint sequential correction requires the original intention filter inputs')
    old_log_prob = data['log_prob'][:, :, i].flatten()
    obs = data['obs'][:, :, i].flatten(0, 1)
    context = data['context'][:, :, i].flatten(0, 1)
    gates = data['gates'][:, :, i].flatten(0, 1)
    actions = data['actions'][:, :, i].flatten(0, 1)
    if not learner.discrete:
        actions = actions[:, :learner.action_dims[i]]
    available = data['available'][:, :, i].flatten(0, 1) if learner.discrete else None
    updated_log_prob = []
    for indices in torch.arange(len(obs), device=learner.device).split(learner.config.minibatch_size):
        current_context = ppo_context(learner, i, context, gates, intention_inputs, indices)
        distribution = agent.distribution(obs[indices], current_context, gates[indices],
                                           available[indices] if available is not None else None)
        joint_log_prob = distribution.log_prob(actions[indices])
        if joint_log_prob.shape != old_log_prob[indices].shape:
            raise ValueError('Sequential PPO requires one joint log probability per rollout row')
        updated_log_prob.append(joint_log_prob)
    log_ratio = torch.cat(updated_log_prob) - old_log_prob
    if learner.config.env == 'smac':
        # A dead predecessor contributes the identity likelihood ratio,
        # irrespective of whether inactive rows were filtered during PPO.
        alive = data['available'][:, :, i, 1:].any(-1).flatten()
        log_ratio = torch.where(alive, log_ratio, torch.zeros_like(log_ratio))
    updated = log_correction.detach() + log_ratio.detach()
    sequential_factor(updated)  # Fail before another agent consumes invalid weights.
    return updated


def update(learner: Learner, data):
    c = learner.config
    # SMAC reserves action zero for dead units; this convention is not shared
    # by the toy task or continuous environments. Keep archived behavior by
    # default, and report the live fraction even when filtering is disabled.
    alive = data['available'][..., 1:].any(-1) if c.env == 'smac' else None
    mask_inactive = c.env == 'smac' and c.mask_inactive_agents
    rewards = learner.tensor(data['rewards'])
    terminated = learner.tensor(data['terminated'], torch.bool)
    truncated = learner.tensor(data['truncated'], torch.bool)
    if c.smac_time_limit_terminal:
        # Optional finite-battle objective: the native cutoff ends the task.
        # Preserve raw environment flags and auxiliary episode boundaries.
        terminated = terminated | truncated
        truncated = torch.zeros_like(truncated)
    advantages, returns = generalized_advantage_estimate(
        rewards, data['values'], data['next_values'],
        terminated, truncated,
        gamma=c.gamma, gae_lambda=c.gae_lambda)
    metrics = []
    sequential = c.update_mode == 'sequential'
    order = torch.randperm(learner.num_agents, device=learner.device).tolist() if sequential else range(learner.num_agents)
    log_correction = torch.zeros_like(data['log_prob'][:, :, 0].flatten()) if sequential else None
    for i in order:
        agent, optimizer = learner.agents[i], learner.optimizers[i]
        intention_inputs = prepare_intention_inputs(learner, data, i)
        full_intention_inputs = intention_inputs
        gradient_parameters = [p for group in optimizer.param_groups for p in group['params']]
        obs = data['obs'][:, :, i].flatten(0, 1)
        contexts = data['context'][:, :, i].flatten(0, 1)
        gates = data['gates'][:, :, i].flatten(0, 1)
        actions = data['actions'][:, :, i].flatten(0, 1)
        if not learner.discrete:
            actions = actions[:, :learner.action_dims[i]]
        old_log_prob = data['log_prob'][:, :, i].flatten()
        available = data['available'][:, :, i].flatten(0, 1) if learner.discrete else None
        advantage, target = advantages[:, :, i].flatten(), returns[:, :, i].flatten()
        old_value = data['values'][:, :, i].flatten()
        if sequential:
            # These indices always refer to the complete [time, environment]
            # grid, not this agent's possibly shorter active minibatch.
            rows = torch.arange(len(obs), device=learner.device)
            factor = sequential_factor(log_correction)
        if mask_inactive:
            active = alive[:, :, i].flatten()
            if not active.any():
                # A zero loss followed by Adam.step() can still move parameters
                # through existing momentum. Skip the optimizer and all moments.
                optimizer.zero_grad(set_to_none=True)
                continue
            if intention_inputs is not None:
                intention_inputs = tuple(value[active] for value in intention_inputs)
            obs, contexts, gates = obs[active], contexts[active], gates[active]
            actions, old_log_prob = actions[active], old_log_prob[active]
            available = available[active]
            advantage, target, old_value = advantage[active], target[active], old_value[active]
            if sequential:
                rows = rows[active]
        # Filter before normalization and minibatching. This also keeps dead
        # rows out of entropy, critic/PopArt targets, and the KL stopping rule.
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
        agent.update_value_scale(target)
        target = (target - agent.value_moments.mean[0]) / agent.value_moments.std[0]
        old_value = (old_value - agent.value_moments.mean[0]) / agent.value_moments.std[0]
        stop = False
        for _ in range(c.epochs):
            for indices in torch.randperm(len(obs), device=learner.device).split(c.minibatch_size):
                context = ppo_context(learner, i, contexts, gates, intention_inputs, indices)
                dist = agent.distribution(obs[indices], context, gates[indices],
                                          available[indices] if available is not None else None)
                log_ratio = dist.log_prob(actions[indices]) - old_log_prob[indices]
                ratio = log_ratio.exp()
                if sequential:
                    # Standardize the local advantage once above; predecessor
                    # ratios weight only the actor's clipped surrogate.
                    surrogate = torch.minimum(ratio * advantage[indices],
                        ratio.clamp(1-c.clip,1+c.clip) * advantage[indices])
                    policy_loss = -(factor[rows[indices]] * surrogate).mean()
                else:
                    policy_loss = -torch.minimum(ratio * advantage[indices],
                        ratio.clamp(1-c.clip,1+c.clip) * advantage[indices]).mean()
                predicted = agent.normalized_value(obs[indices], context, gates[indices])
                clipped = old_value[indices] + (predicted-old_value[indices]).clamp(-c.value_clip,c.value_clip)
                value_loss = 0.5 * torch.maximum((predicted-target[indices]).square(),
                                                 (clipped-target[indices]).square()).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + c.value_coef * value_loss - c.entropy_coef * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite PPO loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                parameters = gradient_parameters if full_intention_inputs is not None else agent.parameters()
                nn.utils.clip_grad_norm_(parameters, c.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                approx_kl = ((ratio - 1) - log_ratio).mean().detach().item()
                metrics.append({'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(),
                                'entropy': entropy.item(), 'approx_kl': approx_kl})
                if c.target_kl > 0 and approx_kl > 1.5*c.target_kl:
                    stop = True
                    break
            if stop:
                break
        if sequential:
            if full_intention_inputs is None:
                log_correction = advance_sequential_correction(learner, data, i, log_correction)
            else:
                log_correction = advance_sequential_correction(
                    learner, data, i, log_correction, full_intention_inputs)
        raw_obs = learner.tensor(data['raw_obs'][:, :, i])
        agent.update_observation_scale(raw_obs.flatten(0, 1)[active] if mask_inactive else raw_obs)
    result = {key: float(np.mean([row[key] for row in metrics])) if metrics else 0.0
              for key in ['policy_loss', 'value_loss', 'entropy', 'approx_kl']}
    if sequential:
        # Diagnostics describe the final cumulative product after all agents,
        # including the last agent's ratio; they do not rescale training weights.
        factor = sequential_factor(log_correction)
        weights = factor.double()
        ess = weights.sum().square() / weights.square().sum()
        result.update(sequential_agent_order=order,
            sequential_factor_min=factor.min().item(), sequential_factor_max=factor.max().item(),
            sequential_factor_ess=ess.item(), sequential_factor_ess_fraction=(ess / len(factor)).item())
    if alive is not None:
        result['alive_fraction'] = alive.float().mean().item()
        result['ppo_update_count'] = len(metrics)
    result['peer_gate_fraction'] = (data['gates'].float().sum().item() /
        max(1, data['gates'].shape[0]*data['gates'].shape[1]*learner.num_agents*(learner.num_agents-1)))
    if learner.latent_dim:
        result['intention_feature_std'] = data['context'].float().std(unbiased=False).item()
    result.update(update_intentions(learner, data))
    return result


def update_intentions(learner: Learner, data):
    """Completed future suffixes stay inside their source episodes."""
    if not learner.latent_dim:
        return {}
    c = learner.config
    length, environments = data['ends'].shape
    candidates = [(t,e) for t in range(length-c.prediction_horizon+1) for e in range(environments)
                  if not data['ends'][t:t+c.prediction_horizon-1,e].any()]
    if not candidates or not c.aux_updates:
        return {}
    summaries, forecasts = [], []
    for update_index in range(c.aux_updates):
        selected = [candidates[j] for j in np.random.randint(len(candidates), size=min(c.aux_batch_size,len(candidates)))]
        ts, es = np.array(selected).T
        for i, (model, optimizer) in enumerate(zip(learner.intentions, learner.aux_optimizers)):
            past = learner.tensor(data['history'][ts,es,:,i])
            visible = learner.tensor(data['visible'][ts,es,:,i], torch.bool)
            future = learner.tensor(np.stack([data['future'][t:t+c.prediction_horizon,e,i] for t,e in selected]))
            future_visible = learner.tensor(np.stack([data['future_visible'][t:t+c.prediction_horizon,e,i] for t,e in selected]),torch.bool)
            if update_index == 0:
                # These suffixes were just collected and have not trained this
                # model. Predict with the past-only prior before any aux step.
                forecasts.append(forecast_diagnostics(model,past,visible,future,future_visible))
            losses = model.loss(past, visible, future, future_visible, beta=c.kl_beta, samples=c.latent_samples)
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('Nonfinite intention objective')
            optimizer.zero_grad(set_to_none=True)
            losses['loss'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            summaries.append({f'intention_{key}': value.detach().item() for key,value in losses.items()})
    result = {key: float(np.mean([row[key] for row in summaries])) for key in summaries[0]}
    result.update(summarize_forecasts(forecasts))
    return result


@torch.no_grad()
def evaluate(learner: Learner, episodes: int, seed: int, *, metadata: dict | None = None):
    """Count completed episodes and optionally record the actual evaluation backend.

    Metadata comes from this rollout environment, not a training manifest or
    the separate environment used to determine policy dimensions.
    """
    from copy import deepcopy

    env = create_env(learner.config)
    records = []
    try:
        if metadata is not None:
            metadata['environment'] = deepcopy(getattr(env, 'metadata', {}))
            metadata['versions'] = {}
            for package in ['torch','numpy','gymnasium','gymnasium-robotics','mujoco',
                            'pysc2','smac','s2clientprotocol','protobuf','gym','mujoco-py']:
                try:
                    metadata['versions'][package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    pass
            metadata['runtime'] = runtime_metadata(learner.config.device)
        for episode in range(episodes):
            state = env.reset(seed=seed+episode)
            if metadata is not None and episode == 0:
                metadata['environment'] = deepcopy(getattr(env, 'metadata', {}))
                metadata['runtime']['initial_environment'] = deepcopy(getattr(env, 'runtime_info', {}))
            history, visible = empty_history(learner, 1)
            episode_return, length = 0.0, 0
            while True:
                inputs = learner.contexts([state], history, visible,
                    learner.completed_episodes >= learner.config.warmup_episodes)
                action, _, _ = learner.action(*inputs, deterministic=True)
                state = env.step(action[0].cpu().numpy())
                advance_history(history, visible, [state])
                episode_return += float(state.rewards.mean())
                length += 1
                if state.episode_done:
                    records.append({'seed': seed+episode,'return':episode_return,'length':length,
                                    'battle_won':state.info.get('battle_won')})
                    break
    finally:
        try:
            if metadata is not None:
                metadata.setdefault('runtime', {})['final_environment'] = deepcopy(getattr(env, 'runtime_info', {}))
        finally:
            env.close()
    return records


def run(config: Config):
    config.validate()
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'metrics.jsonl').exists():
        raise FileExistsError(f'{output} already contains a run; choose a fresh output directory')
    torch.set_num_threads(config.torch_threads)
    seed_everything(config.seed)
    environments, executor = [], None
    steps, episodes = 0, 0
    try:
        for _ in range(config.num_envs):
            environments.append(create_env(config))
        states = [env.reset(seed=config.seed+1000*e) for e,env in enumerate(environments)]
        learner = Learner(config,environments[0])
        versions = {}
        for package in ['torch','numpy','gymnasium','gymnasium-robotics','mujoco',
                        'pysc2','smac','s2clientprotocol','protobuf','gym','mujoco-py']:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        (output/'config.json').write_text(json.dumps(asdict(config),indent=2)+'\n')
        shutil.copytree(Path(__file__).parent, output/'source'/'dms',
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        source_hashes = {str(path.relative_to(output/'source')):hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted((output/'source').rglob('*.py'))}
        (output/'manifest.json').write_text(json.dumps({'versions':versions,
            'environment':getattr(environments[0],'metadata',{}),'initial_info':states[0].info,
            'source_sha256':source_hashes,
            'runtime':runtime_metadata(config.device),
            'information':'local observations and visibility-masked observed action sensors',
            'intention_control':'deterministic prior means; fixed rollout features/gates during PPO',
            'value_time_limits':{
                'mode':'terminal_at_smac_limit' if config.smac_time_limit_terminal else 'bootstrap_truncations',
                'native_environment_flags_preserved':True,
                'scope':'value/advantage targets only; rewards, resets, observations and intention targets unchanged'},
            'ppo_update':{'mode':config.update_mode,
                'agent_order':'one random permutation per rollout' if config.update_mode == 'sequential' else 'fixed index order; independent objectives',
                'training_signal':'detached predecessor action likelihood ratios across agents' if config.update_mode == 'sequential' else 'no cross-agent likelihood correction',
                'actor_information':'unchanged local cached observations, intentions and gates',
                'sequential_scope':'HAPPO-style coordinated training; not the exact legacy stochastic-latent recipe'},
            'parameters':{'actor_critic':sum(p.numel() for p in learner.agents.parameters()),
                          'intention':sum(p.numel() for p in learner.intentions.parameters())},
            'history':'finite rolling action window; configurable redesign of full-history prior',
            'status':'training; no paper performance claim'},indent=2,default=str)+'\n')
        history,visible = empty_history(learner,config.num_envs)
        episode_returns = np.zeros(config.num_envs)
        steps,episodes,next_eval = 0,0,config.eval_every
        training_battles, training_battles_won = 0, 0
        if config.parallel_env_steps:
            executor = ThreadPoolExecutor(max_workers=config.num_envs)
        started = time.monotonic()
        while steps < config.total_steps:
            rollout_steps = min(config.rollout_steps, (config.total_steps-steps)//config.num_envs)
            data,states,history,visible,episodes,episode_returns,finished = collect(
                learner,environments,states,history,visible,episodes,episode_returns,rollout_steps,executor)
            metrics = update(learner,data)
            learner.completed_episodes = episodes
            steps += rollout_steps*config.num_envs
            metrics.update(steps=steps,episodes=episodes,elapsed_seconds=time.monotonic()-started,
                           training_return_mean=float(np.mean([r['return'] for r in finished])) if finished else None)
            if config.env == 'smac':
                outcomes = [r['battle_won'] for r in finished if r['battle_won'] is not None]
                wins = sum(bool(won) for won in outcomes)
                training_battles += len(outcomes)
                training_battles_won += wins
                metrics.update(training_battles=len(outcomes), training_battles_won=wins,
                    training_win_rate=wins/len(outcomes) if outcomes else None,
                    training_battles_total=training_battles,
                    training_battles_won_total=training_battles_won,
                    training_win_rate_total=training_battles_won/training_battles if training_battles else None)
                metrics['scheduled_environment_restarts'] = sum(
                    env.runtime_info['scheduled_restarts'] for env in environments)
            append_json(output/'metrics.jsonl',metrics)
            print(json.dumps(metrics),flush=True)
            if steps >= next_eval or steps >= config.total_steps:
                # Evaluation should not perturb training's pseudo-random streams.
                rng = (random.getstate(),np.random.get_state(),torch.get_rng_state(),
                       torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
                results = evaluate(learner,config.eval_episodes,config.eval_seed)
                random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
                if rng[3] is not None:
                    torch.cuda.set_rng_state_all(rng[3])
                for result in results:
                    append_json(output/'evaluation.jsonl',{'steps':steps,**result})
                summary = {'steps':steps,'evaluation_return_mean':float(np.mean([r['return'] for r in results]))}
                if all(r['battle_won'] is not None for r in results):
                    summary['evaluation_win_rate'] = float(np.mean([r['battle_won'] for r in results]))
                print(json.dumps(summary),flush=True)
                learner.save(output/'latest.pt',steps,episodes)
                next_eval += config.eval_every
        manifest = json.loads((output/'manifest.json').read_text())
        manifest.update(status='completed; no paper performance claim',
                        completed_steps=steps,completed_episodes=episodes,
                        environment_runtime=[getattr(env,'runtime_info',{}) for env in environments])
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return output
    except BaseException as error:
        manifest_path = output/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest.update(status='failed; no paper performance claim',
                            completed_steps=steps,completed_episodes=episodes,
                            failure={'type':type(error).__name__,'message':str(error)},
                            environment_runtime=[getattr(env,'runtime_info',{}) for env in environments])
            manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        for env in environments:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--device')
    parser.add_argument('--output')
    parser.add_argument('--total-steps',type=int)
    parser.add_argument('--algorithm',choices=['ippo','dms'])
    args = parser.parse_args()
    values = json.loads(args.config.read_text()) if args.config else {}
    known = {field.name for field in fields(Config)}
    if set(values)-known:
        parser.error(f'Unknown configuration keys: {set(values)-known}')
    for name in ['seed','device','output','total_steps','algorithm']:
        value = getattr(args,name)
        if value is not None:
            values[name] = value
    run(Config(**values))


if __name__ == '__main__':
    main()
