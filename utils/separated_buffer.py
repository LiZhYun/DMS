import torch
import numpy as np
from collections import defaultdict
from utils.util import check, get_shape_from_obs_space, get_shape_from_act_space
from torch.distributions import kl_divergence, Normal
import copy 
import torch.nn as nn 
import torch.nn.functional as F

def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1,0,2).reshape(-1, *x.shape[2:])

class SeparatedReplayBuffer(object):
    def __init__(self, args, obs_space, share_obs_space, act_space, n_agents, action_dim, train_share_observation_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.rnn_hidden_size = args.hidden_size
        self.rnn_intention_hidden_size = args.intention_hidden_size
        self.num_sample = args.num_sample
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.hidden_size = args.hidden_size
        self.n_agents = n_agents         
        self.args = args         
        self.action_dim = action_dim  

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        train_share_obs_shape = get_shape_from_obs_space(train_share_observation_space)

        self.intention_size = train_share_obs_shape[0]
        if self.args.cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *obs_shape), dtype=np.float32)
        self.train_share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *train_share_obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        # self.intentions = torch.zeros(self.episode_length + 1, self.n_rollout_threads, self.n_agents, self.intention_size, device=torch.device('cpu'))
        # self.critic_obs = torch.zeros(self.episode_length + 1, self.n_rollout_threads, self.hidden_size, device=torch.device(self.device))
        # self.critic_states = torch.zeros(self.episode_length + 1, self.n_rollout_threads, self.hidden_size, device=torch.device(self.device))

        self.intentions = np.random.normal(0, 1, size=(self.episode_length + 1, self.n_rollout_threads, self.n_agents, self.intention_size))
        # self.intentions = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.n_agents, self.intention_size), dtype=np.float32)
        self.correlated_agents = np.ones((self.episode_length + 1, self.n_rollout_threads, self.n_agents), dtype=np.float32)
        self.rnn_states_intention = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.n_agents, self.recurrent_N, self.rnn_intention_hidden_size), dtype=np.float32)  
        self.last_actions = np.zeros((self.episode_length, self.n_rollout_threads, self.n_agents, self.action_dim), dtype=np.float32)
        

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, act_space.n), dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.factor = None

        self.step = 0

    def update_factor(self, factor):
        self.factor = factor.copy()
    
    def update_action(self, last_actions):
        tmp = self.last_actions == 0.
        self.last_actions = last_actions.copy()
        self.last_actions[tmp == True] = np.zeros(((tmp == True).sum()), dtype=np.float32)

    # def update_action(self, last_actions, randomseq):
    #     if len(randomseq) > 0:
    #         tmp = last_actions[:, :, randomseq].copy() == 0.
    #         self.last_actions[:, :, randomseq] = last_actions[:, :, randomseq].copy()
    #         self.last_actions[:, :, randomseq][tmp == True] = np.zeros(((tmp == True).sum()), dtype=np.float32)

    def insert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, rnn_states_intention, correlated_agents, all_last_actions, bad_masks=None, active_masks=None, available_actions=None, train_share_obs=None):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        # self.intentions[self.step + 1] = intention.clone()
        self.train_share_obs[self.step + 1] = train_share_obs.copy()
        # self.critic_states[self.step + 1] = critic_states.copy()
        self.correlated_agents[self.step + 1] = correlated_agents.copy()
        self.rnn_states_intention[self.step + 1] = rnn_states_intention.copy()
        self.last_actions[self.step] = all_last_actions.copy()
    
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length
    
    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()

        self.intentions[0] = self.intentions[-1].copy()      
        self.train_share_obs[0] = self.train_share_obs[-1].copy()
        # self.critic_states[0] = self.critic_states[-1].copy()
        self.correlated_agents[0] = self.correlated_agents[-1].copy()         
        self.rnn_states_intention[0] = self.rnn_states_intention[-1].copy()
        self.last_actions[0] = self.last_actions[-1].copy()

        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        use proper time limits, the difference of use or not is whether use bad_mask
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[
                            step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        if self.factor is not None:
            # factor = self.factor.reshape(-1,1)
            factor = self.factor.reshape(-1, self.factor.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            if self.factor is None:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch
            else:
                factor_batch = factor[indices]
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        n_rollout_threads = self.rewards.shape[1]
        assert n_rollout_threads >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_mini_batch))
        num_envs_per_batch = n_rollout_threads // num_mini_batch
        perm = torch.randperm(n_rollout_threads).numpy()
        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            factor_batch = []
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(self.share_obs[:-1, ind])
                obs_batch.append(self.obs[:-1, ind])
                rnn_states_batch.append(self.rnn_states[0:1, ind])
                rnn_states_critic_batch.append(self.rnn_states_critic[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(self.available_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                active_masks_batch.append(self.active_masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                if self.factor is not None:
                    factor_batch.append(self.factor[:,ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            if self.factor is not None:
                factor_batch=np.stack(factor_batch,1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, -1) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch, 1).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch, 1).reshape(N, *self.rnn_states_critic.shape[2:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch=_flatten(T,N,factor_batch)
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)
            if self.factor is not None:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch
            else:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length, encoder_decoder, long_short_clip=False, episode=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length  # [C=r*T/L]
        mini_batch_size = data_chunks // num_mini_batch

        assert episode_length * n_rollout_threads >= data_chunk_length, (
            "PPO requires the number of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, episode_length, data_chunk_length))
        assert data_chunks >= 2, ("need larger batch size")

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        min_batch_size_last_actions = episode_length // num_mini_batch

        if len(self.share_obs.shape) > 3:
            share_obs = self.share_obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.share_obs.shape[2:])
            obs = self.obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])

        # intentions = self.intentions[:-1].permute(1, 0, 2, 3).reshape(-1, *self.intentions.shape[2:])
        intentions = self.intentions[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.intentions.shape[2:])
        # critic_obs = _cast(self.critic_obs[:-1])
        # critic_states = _cast(self.critic_states[:-1])
        correlated_agents = _cast(self.correlated_agents[:-1])
        last_actions_all = self.last_actions.transpose(1, 0, 2, 3).reshape(-1, *self.last_actions.shape[2:])
        # last_actions_all = self.last_actions.transpose(1, 2, 3, 0) # 32, 3, 1, 200
        rnn_states_intention = self.rnn_states_intention[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.rnn_states_intention.shape[2:])
        

        if self.factor is not None:
            factor = _cast(self.factor)
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states_critic.shape[2:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        action_batch_idx = 0
        for sid, indices in enumerate(sampler):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            factor_batch = []

            intentions_batch = []
            # critic_obs_batch = []
            # critic_states_batch = []
            last_actions_batch = []
            correlated_agents_batch = []
            rnn_states_intention_batch = []
            rnn_states_train_intention_batch = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind+data_chunk_length])
                obs_batch.append(obs[ind:ind+data_chunk_length])
                actions_batch.append(actions[ind:ind+data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind+data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind+data_chunk_length])
                return_batch.append(returns[ind:ind+data_chunk_length])
                masks_batch.append(masks[ind:ind+data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind+data_chunk_length])
                adv_targ.append(advantages[ind:ind+data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

                intentions_batch.append(intentions[ind:ind+data_chunk_length])
                # critic_obs_batch.append(critic_obs[ind:ind+data_chunk_length])
                # critic_states_batch.append(critic_states[ind:ind+data_chunk_length])
                last_actions_batch.append(last_actions_all[ind:ind+data_chunk_length])
                correlated_agents_batch.append(correlated_agents[ind:ind+data_chunk_length])
                rnn_states_intention_batch.append(rnn_states_intention[ind]) #todo
                # rnn_states_intention_batch.append(rnn_states_intention[ind:ind+data_chunk_length]) #todo
 
                if self.factor is not None:
                    factor_batch.append(factor[ind:ind+data_chunk_length])

            # last_actions_batch = last_actions_all[:,:,:,action_batch_idx * min_batch_size_last_actions:(action_batch_idx+1)*min_batch_size_last_actions] 
            # action_rand = torch.randperm(last_actions_batch.shape[0]).numpy()               
            # last_actions_batch = last_actions_batch[action_rand]
            # rnn_states_train_intention_batch = self.rnn_states_intention[:-1][action_batch_idx * min_batch_size_last_actions]
            # rnn_states_train_intention_batch = rnn_states_train_intention_batch[action_rand]
            # action_batch_idx = (action_batch_idx + 1) % 10

            # last_actions_batch = last_actions_all[:,:,:,sid*min_batch_size_last_actions:(sid+1)*min_batch_size_last_actions]   
            # action_rand = torch.randperm(last_actions_batch.shape[0]).numpy()     
            # last_actions_batch = last_actions_batch[action_rand]       
            # rnn_states_train_intention_batch = self.rnn_states_intention[:-1][sid*min_batch_size_last_actions]
            # rnn_states_train_intention_batch = rnn_states_train_intention_batch[action_rand]

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (N, L, Dim)
            share_obs_batch = np.stack(share_obs_batch)
            obs_batch = np.stack(obs_batch)

            # intentions_batch = torch.stack(intentions_batch)
            intentions_batch = np.stack(intentions_batch)
            # critic_obs_batch = np.stack(critic_obs_batch)
            # critic_states_batch = np.stack(critic_states_batch)
            last_actions_batch = np.stack(last_actions_batch)
            correlated_agents_batch = np.stack(correlated_agents_batch)
            rnn_states_intention_batch = np.stack(rnn_states_intention_batch).reshape(N, *self.rnn_states_intention.shape[2:])
            # rnn_states_intention_batch = np.stack(rnn_states_intention_batch).reshape(L*N, *self.rnn_states_intention.shape[2:])

            actions_batch = np.stack(actions_batch)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch)
            if self.factor is not None:
                factor_batch = np.stack(factor_batch)
            value_preds_batch = np.stack(value_preds_batch)
            return_batch = np.stack(return_batch)
            masks_batch = np.stack(masks_batch)
            active_masks_batch = np.stack(active_masks_batch)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch)
            adv_targ = np.stack(adv_targ)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[2:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)

            intentions_batch = _flatten(L, N, intentions_batch)
            # critic_obs_batch = _flatten(L, N, critic_obs_batch)
            # critic_states_batch = _flatten(L, N, critic_states_batch)
            last_actions_batch = _flatten(L, N, last_actions_batch)
            correlated_agents_batch = _flatten(L, N, correlated_agents_batch)

            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(L, N, factor_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            # mini_length = last_actions_batch.shape[-1]             
            # long_short_step = 4 
            # if long_short_clip and ((episode + 1) % self.args.long_short_clip_freq) == 0:
            #     last_action_clipped = copy.deepcopy(last_actions_batch)
                
            #     for env_idx in range(last_actions_batch.shape[0]):
            #         for agent_idx in range(last_actions_batch.shape[1]):
            #             for idx in range(0, mini_length, long_short_step): # 16, 2, 4, 25
            #                 temp_actions = check(last_action_clipped[env_idx, agent_idx,:,idx:idx+long_short_step]).to(**self.tpdv).clone() # 16, 2, 4, 3
            #                 temp_actions = temp_actions.reshape(1, *temp_actions.shape)
            #                 # temp_actions= temp_actions.permute(0,1,3,2) # 16, 2, 3, 4
            #                 temp_actions= temp_actions.permute(0,2,1) # 16, 2, 3, 4
            #                 # temp_actions = temp_actions.reshape(-1, long_short_step, self.action_dim) # 32, 3, 4
            #                 intention, _ = encoder_decoder.encoder(temp_actions) # Normal((1, 20), (1, 20))
            #                 # intention_buffer_mean = check(intention_buffer.mean).to(**self.tpdv)    # 6, 20 -> 2, 3, 20 -> 2, 20 - > 1, 20
            #                 # intention_buffer_scale = check(intention_buffer.scale).to(**self.tpdv) # 6, 20 -> 2, 3, 20 -> 2, 20 - > 1, 20
            #                 # kl_ratio = kl_divergence(intention, Normal(torch.from_numpy(intention_buffer)[env_idx].clone().reshape(1, *intention_buffer.shape[1:]).view(-1, intention_buffer.shape[-1]).to(**self.tpdv), torch.ones_like(torch.from_numpy(intention_buffer)[env_idx].clone().reshape(1, *intention_buffer.shape[1:]).view(-1, intention_buffer.shape[-1])).to(**self.tpdv))).mean()
            #                 KLDivLoss = nn.KLDivLoss(reduction='batchmean')             
            #                 p_output = F.softmax(intention.mean, dim=-1)
            #                 q_output = F.softmax(torch.from_numpy(self.intentions[-1])[env_idx, agent_idx].clone().reshape(1, *self.intentions[-1].shape[2:]).view(-1, self.intentions[-1].shape[-1]).to(**self.tpdv), dim=-1)     
            #                 log_mean_output = ((p_output + q_output )/2).log()     
            #                 kl_ratio = (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2
            #                 if kl_ratio > self.args.long_short_coef:
            #                     last_action_clipped[env_idx,agent_idx,:,idx:idx+long_short_step] = np.zeros((*last_action_clipped.shape[2:-1], long_short_step))

            #     last_actions = last_action_clipped.transpose(0, 3, 1, 2) # 32, 200, 3, 1
            #     past_len = int(mini_length) - 20
            #     last_actions_past = last_actions[:, :past_len]
            #     last_actions_future = last_actions[:, past_len:]
            # else:
            # last_actions = copy.deepcopy(last_actions_batch)
            # # last_actions = last_actions.reshape(*last_actions.shape, -1) # 32, 3, 200, 1
            # last_actions = last_actions.transpose(0, 3, 1, 2) # # 32, 200, 3, 1
            # past_len =  episode_length - 20
            # last_actions_past = last_actions[:, :past_len]
            # last_actions_future = last_actions[:, past_len:]

            if self.factor is not None:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch, last_actions_batch, intentions_batch, correlated_agents_batch, rnn_states_intention_batch, rnn_states_train_intention_batch
            else:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def intention_recurrent_generator(self, batch_size, encoder_decoder, long_short_clip=False, episode=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]

        # rand = list(range(n_rollout_threads))
        rand = torch.randperm(n_rollout_threads).numpy()

        sampler = [rand[i*batch_size:(i+1)*batch_size] for i in range(n_rollout_threads // batch_size)]

        min_batch_size_last_actions = batch_size
        
        last_actions_all = self.last_actions.transpose(1, 2, 3, 0) # 32, 3, 1, 200

        action_batch_idx = 0
        for indices in sampler:
            masks_batch = []
            critic_obs_batch = []
            critic_states_batch = []
            rnn_states_intention_batch = []
            last_actions_batch = []
            rnn_states_train_intention_batch = []
            rnn_states_critic_batch = []
            intentions_batch = []
            correlated_agents_batch = []
            for index in indices:
                masks_batch.append(self.masks[:-1, index])
                last_actions_batch.append(last_actions_all[index])
                rnn_states_train_intention_batch.append(self.rnn_states_intention[:-1][0, index])
                rnn_states_intention_batch.append(self.rnn_states_intention[:-1, index])
                critic_obs_batch.append(self.obs[:-1, index])
                critic_states_batch.append(self.train_share_obs[:-1, index])
                rnn_states_critic_batch.append(self.rnn_states_critic[:-1, index])
                intentions_batch.append(self.intentions[:-1, index])
                correlated_agents_batch.append(self.correlated_agents[:-1, index])
            last_actions_batch = np.stack(last_actions_batch)
            rnn_states_train_intention_batch = np.stack(rnn_states_train_intention_batch)
            rnn_states_intention_batch = np.stack(rnn_states_intention_batch)
            critic_obs_batch = np.stack(critic_obs_batch)
            critic_states_batch = np.stack(critic_states_batch)
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch)
            masks_batch = np.stack(masks_batch)
            intentions_batch = np.stack(intentions_batch)
            correlated_agents_batch = np.stack(correlated_agents_batch)

            # last_actions_batch = last_actions_all[sid*min_batch_size_last_actions:(sid+1)*min_batch_size_last_actions,:,:,:]   
            # action_rand = torch.randperm(last_actions_batch.shape[0]).numpy()     
            # last_actions_batch = last_actions_batch[action_rand]       
            # rnn_states_train_intention_batch = self.rnn_states_intention[:-1][0, sid*min_batch_size_last_actions:(sid+1)*min_batch_size_last_actions]             # rnn_states_train_intention_batch = rnn_states_train_intention_batch[action_rand]
            mini_length = last_actions_batch.shape[-1]             
            long_short_step = 20 
            if long_short_clip and ((episode + 1) % self.args.long_short_clip_freq) == 0:
                last_action_clipped = copy.deepcopy(last_actions_batch)
                with torch.no_grad():
                    graph = encoder_decoder.build_input(last_action_clipped.transpose(0, 3, 1, 2))
                    # temp_actions = check(last_action_clipped[:, agent_idx,:,idx:idx+long_short_step]).to(**self.tpdv).clone() # 16, 2, 4, 3
                    # temp_actions = temp_actions.reshape(last_actions_batch.shape[0], *temp_actions.shape)
                    # temp_actions= temp_actions.permute(0,2,1) # 16, 2, 3, 4
                    obs = check(critic_obs_batch).to(**self.tpdv)
                    rnn_states = check(rnn_states_critic_batch[:, 0:1]).to(**self.tpdv).reshape(-1, *self.rnn_states.shape[2:])         
                    rnn_states_intention = check(rnn_states_intention_batch[:, 0:1]).to(**self.tpdv).reshape(-1, self.recurrent_N, self.rnn_intention_hidden_size).transpose(0, 1)         
                    masks = check(masks_batch).to(**self.tpdv)         
                    correlated_agents = check(correlated_agents_batch).to(**self.tpdv)
                    # rnn_states_intention = rnn_states_intention # batchsize * n_agents, 1, 64 -> 1, batchsize * n_agents, 64 
                    q_res = encoder_decoder.get_intention(graph, rnn_states_intention, observ=obs, correlated_agents=correlated_agents, rnn_states=rnn_states, masks=masks, forecast=True) 
                    long_intention = q_res["intention"].mean(1) 
                    # for env_idx in range(last_actions_batch.shape[0]):
                    # for agent_idx in range(last_actions_batch.shape[1]):
                    for idx in range(0, mini_length, long_short_step): # 16, 2, 4, 25
                        graph = encoder_decoder.build_input(last_action_clipped[:,:,:,idx:idx+long_short_step].transpose(0, 3, 1, 2))
                        # temp_actions = check(last_action_clipped[:, agent_idx,:,idx:idx+long_short_step]).to(**self.tpdv).clone() # 16, 2, 4, 3
                        # temp_actions = temp_actions.reshape(last_actions_batch.shape[0], *temp_actions.shape)
                        # temp_actions= temp_actions.permute(0,2,1) # 16, 2, 3, 4
                        obs = check(critic_obs_batch[:, idx:idx+long_short_step]).to(**self.tpdv)
                        rnn_states = check(rnn_states_critic_batch[:, idx]).to(**self.tpdv)         
                        rnn_states_intention = check(rnn_states_intention_batch[:, idx]).to(**self.tpdv).reshape(-1, self.recurrent_N, self.rnn_intention_hidden_size).transpose(0, 1)         
                        masks = check(masks_batch[:, idx:idx+long_short_step]).to(**self.tpdv)         
                        correlated_agents = check(correlated_agents_batch[:, idx:idx+long_short_step]).to(**self.tpdv)
                        # rnn_states_intention = rnn_states_intention.reshape(-1, self.recurrent_N, self.rnn_intention_hidden_size).transpose(0, 1) # batchsize * n_agents, 1, 64 -> 1, batchsize * n_agents, 64 
                        q_res = encoder_decoder.get_intention(graph, rnn_states_intention, observ=obs, correlated_agents=correlated_agents, rnn_states=rnn_states, masks=masks, forecast=True) 
                        short_intention = q_res["intention"].mean(1) 
                        # actions = graph.ndata['feat']
                        # zIG_rv, zIA_rv, _ = encoder_decoder.encoder(graph, actions)
                        # zIA = zIA_rv.rsample([self.num_sample]).transpose(1, 0)
                        # zIG = zIG_rv.rsample([self.num_sample]).transpose(1, 0)
                        # graph.ndata['zIA'] = zIA
                        # graph.ndata['zIG'] = zIG
                        # intention = encoder_decoder.decoder(graph, graph.ndata['feat'], forecast=True).mean(1).reshape(batch_size, self.n_agents, self.intention_size)
                        # q_res = self.policy.get_intention(graph, None, True)
                        # intentions[:, agent_id] = _t2n(q_res["intention"]).reshape(self.n_rollout_threads, self.num_agents, self.intention_size)
                        # intention = q_res["intention"]

                        # temp_intention = Normal(torch.from_numpy(self.intentions[-1, sid*min_batch_size_last_actions:(sid+1)*min_batch_size_last_actions, agent_idx]).clone().to(**self.tpdv), torch.tensor([1e-5], device=torch.device('cuda')).repeat(*self.intentions[-1, sid*min_batch_size_last_actions:(sid+1)*min_batch_size_last_actions, agent_idx].shape))
                        # temp_intention = []
                        # for index in indices:
                        #     temp_intention.append(self.intentions[-1, index])
                        # temp_intention = np.stack(temp_intention)
                        # kl_ratio = kl_divergence(intention, Normal(torch.from_numpy(self.intentions[-1, sid*min_batch_size_last_actions:(sid+1)*min_batch_size_last_actions, agent_idx]).clone().to(**self.tpdv), torch.tensor([1e-5], device=torch.device('cuda')).repeat(*self.intentions[-1, sid*min_batch_size_last_actions:(sid+1)*min_batch_size_last_actions, agent_idx].shape)))
                        kl_ratio = kl_divergence(Normal(short_intention, torch.tensor([1e-5], device=torch.device(self.device)).repeat(short_intention.shape)), Normal(long_intention, torch.tensor([1e-5], device=torch.device(self.device)).repeat(long_intention.shape)))

                        # KLDivLoss = nn.KLDivLoss(reduction='batchmean')             
                        # p_output = F.softmax(intention.mean, dim=-1)
                        # q_output = F.softmax(torch.from_numpy(self.intentions[-1])[env_idx, agent_idx].clone().reshape(1, *self.intentions[-1].shape[2:]).view(-1, self.intentions[-1].shape[-1]).to(**self.tpdv), dim=-1)     
                        # log_mean_output = ((p_output + q_output )/2).log()     
                        # kl_ratio = (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2
                        kl_ratio = kl_ratio.mean(dim=-1).view(-1, 1, 1, 1).repeat(1, self.n_agents,last_action_clipped.shape[2], long_short_step).cpu().numpy()
                        last_action_clipped[:,:,:,idx:idx+long_short_step] = np.where(kl_ratio > self.args.long_short_coef, np.zeros((last_actions_batch.shape[0], last_actions_batch.shape[1], last_action_clipped.shape[2], long_short_step)), last_action_clipped[:,:,:,idx:idx+long_short_step])
                            # if kl_ratio > self.args.long_short_coef:
                            #     last_action_clipped[:,agent_idx,:,idx:idx+long_short_step] = np.zeros((*last_action_clipped.shape[2:-1], long_short_step))

                last_actions = last_action_clipped.transpose(0, 3, 1, 2) # 32, 200, 3, 1
            else:
                last_actions = copy.deepcopy(last_actions_batch)
                last_actions = last_actions.transpose(0, 3, 1, 2) # # 32, 200, 3, 1
    
            # past_len = 10
            # past_len = int(mini_length) - 90  
            # last_actions_past = last_actions[:, :past_len]                 
            # last_actions_future = last_actions[:, past_len:]
            yield last_actions, rnn_states_train_intention_batch, rnn_states_intention_batch, critic_obs_batch, critic_states_batch, rnn_states_critic_batch, masks_batch, intentions_batch, correlated_agents_batch

    # def intention_recurrent_generator(self, batch_size, encoder_decoder, long_short_clip=False, episode=None):
    #     episode_length, n_rollout_threads = self.rewards.shape[0:2]

    #     rand = torch.randperm(n_rollout_threads // batch_size).numpy()
    #     # sampler = [rand[i*train_len:(i+1)*train_len] for i in range(episode_length // train_len)]

    #     min_batch_size_last_actions = batch_size
        
    #     last_actions_all = self.last_actions.transpose(1, 2, 3, 0) # 32, 3, 1, 200

    #     action_batch_idx = 0
    #     for sid in rand:
    #         last_actions_batch = last_actions_all[sid*min_batch_size_last_actions:(sid+1)*min_batch_size_last_actions,:,:,:]   
    #         # action_rand = torch.randperm(last_actions_batch.shape[0]).numpy()     
    #         # last_actions_batch = last_actions_batch[action_rand]       
    #         rnn_states_train_intention_batch = self.rnn_states_intention[:-1][0, sid*min_batch_size_last_actions:(sid+1)*min_batch_size_last_actions]
    #         # rnn_states_train_intention_batch = rnn_states_train_intention_batch[action_rand]
    #         mini_length = last_actions_batch.shape[-1]             
    #         long_short_step = 4 
    #         if long_short_clip and ((episode + 1) % self.args.long_short_clip_freq) == 0:
    #             last_action_clipped = copy.deepcopy(last_actions_batch)
                
    #             for env_idx in range(last_actions_batch.shape[0]):
    #                 for agent_idx in range(last_actions_batch.shape[1]):
    #                     for idx in range(0, mini_length, long_short_step): # 16, 2, 4, 25
    #                         temp_actions = check(last_action_clipped[env_idx, agent_idx,:,idx:idx+long_short_step]).to(**self.tpdv).clone() # 16, 2, 4, 3
    #                         temp_actions = temp_actions.reshape(1, *temp_actions.shape)
    #                         temp_actions= temp_actions.permute(0,2,1) # 16, 2, 3, 4
    #                         intention, _ = encoder_decoder.encoder(temp_actions) # Normal((1, 20), (1, 20))
    #                         KLDivLoss = nn.KLDivLoss(reduction='batchmean')             
    #                         p_output = F.softmax(intention.mean, dim=-1)
    #                         q_output = F.softmax(torch.from_numpy(self.intentions[-1])[env_idx, agent_idx].clone().reshape(1, *self.intentions[-1].shape[2:]).view(-1, self.intentions[-1].shape[-1]).to(**self.tpdv), dim=-1)     
    #                         log_mean_output = ((p_output + q_output )/2).log()     
    #                         kl_ratio = (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2
    #                         if kl_ratio > self.args.long_short_coef:
    #                             last_action_clipped[env_idx,agent_idx,:,idx:idx+long_short_step] = np.zeros((*last_action_clipped.shape[2:-1], long_short_step))

    #             last_actions = last_action_clipped.transpose(0, 3, 1, 2) # 32, 200, 3, 1
    #             past_len = int(mini_length) - 20
    #             last_actions_past = last_actions[:, :past_len]
    #             last_actions_future = last_actions[:, past_len:]
    #         else:
    #             last_actions = copy.deepcopy(last_actions_batch)
    #             last_actions = last_actions.transpose(0, 3, 1, 2) # # 32, 200, 3, 1
    #             past_len =  int(mini_length) - 20
    #             last_actions_past = last_actions[:, :past_len]
    #             last_actions_future = last_actions[:, past_len:]

    #         yield last_actions_past, last_actions_future, rnn_states_train_intention_batch


    # def intention_recurrent_generator(self, train_len, encoder_decoder, long_short_clip=False, episode=None):
    #     episode_length, n_rollout_threads = self.rewards.shape[0:2]

    #     rand = torch.randperm(episode_length // train_len).numpy()
    #     # sampler = [rand[i*train_len:(i+1)*train_len] for i in range(episode_length // train_len)]

    #     min_batch_size_last_actions = train_len

    #     last_actions_all = self.last_actions.transpose(1, 2, 3, 0) # 32, 3, 1, 200

    #     action_batch_idx = 0
    #     for sid in rand:
    #         last_actions_batch = last_actions_all[:,:,:,sid*min_batch_size_last_actions:(sid+1)*min_batch_size_last_actions]   
    #         action_rand = torch.randperm(last_actions_batch.shape[0]).numpy()     
    #         last_actions_batch = last_actions_batch[action_rand]       
    #         rnn_states_train_intention_batch = self.rnn_states_intention[:-1][sid*min_batch_size_last_actions]
    #         rnn_states_train_intention_batch = rnn_states_train_intention_batch[action_rand]
    #         mini_length = last_actions_batch.shape[-1]             
    #         long_short_step = 4 
    #         if long_short_clip and ((episode + 1) % self.args.long_short_clip_freq) == 0:
    #             last_action_clipped = copy.deepcopy(last_actions_batch)
                
    #             for env_idx in range(last_actions_batch.shape[0]):
    #                 for agent_idx in range(last_actions_batch.shape[1]):
    #                     for idx in range(0, mini_length, long_short_step): # 16, 2, 4, 25
    #                         temp_actions = check(last_action_clipped[env_idx, agent_idx,:,idx:idx+long_short_step]).to(**self.tpdv).clone() # 16, 2, 4, 3
    #                         temp_actions = temp_actions.reshape(1, *temp_actions.shape)
    #                         temp_actions= temp_actions.permute(0,2,1) # 16, 2, 3, 4
    #                         intention, _ = encoder_decoder.encoder(temp_actions) # Normal((1, 20), (1, 20))
    #                         KLDivLoss = nn.KLDivLoss(reduction='batchmean')             
    #                         p_output = F.softmax(intention.mean, dim=-1)
    #                         q_output = F.softmax(torch.from_numpy(self.intentions[-1])[env_idx, agent_idx].clone().reshape(1, *self.intentions[-1].shape[2:]).view(-1, self.intentions[-1].shape[-1]).to(**self.tpdv), dim=-1)     
    #                         log_mean_output = ((p_output + q_output )/2).log()     
    #                         kl_ratio = (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2
    #                         if kl_ratio > self.args.long_short_coef:
    #                             last_action_clipped[env_idx,agent_idx,:,idx:idx+long_short_step] = np.zeros((*last_action_clipped.shape[2:-1], long_short_step))

    #             last_actions = last_action_clipped.transpose(0, 3, 1, 2) # 32, 200, 3, 1
    #             past_len = int(mini_length) - 5
    #             last_actions_past = last_actions[:, :past_len]
    #             last_actions_future = last_actions[:, past_len:]
    #         else:
    #             last_actions = copy.deepcopy(last_actions_batch)
    #             last_actions = last_actions.transpose(0, 3, 1, 2) # # 32, 200, 3, 1
    #             past_len =  int(mini_length) - 5
    #             last_actions_past = last_actions[:, :past_len]
    #             last_actions_future = last_actions[:, past_len:]

    #         yield last_actions_past, last_actions_future, rnn_states_train_intention_batch