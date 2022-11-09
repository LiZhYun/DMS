import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.popart import PopArt
from algorithms.utils.util import check
from torch.distributions import Normal, kl_divergence

class HAPPO():
    """
    Trainer class for HAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (HAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents, state_size,
                 device=torch.device("cpu"),
                 buffer=None):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.state_size = state_size

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        self.tau = args.intention_update_beta
        self.num_agents = num_agents
        self.args = args         
        # self.intention_size = args.intention_size
        self.intention_size = self.args.intention_size
        self.hidden_size = args.hidden_size
        self.intention_hidden_size = args.intention_hidden_size
        self.intention_update_freq = args.intention_update_freq
        self.actor_update_freq = args.actor_update_freq         
        self.critic_update_freq = args.critic_update_freq
        self.episode_length = args.episode_length
        self.preupdate = args.preupdate
        self.env_name = args.env_name
        self.n_rollout_threads = args.n_rollout_threads         
        self.recurrent_N = args.recurrent_N
        self.use_intention = args.use_intention
        # self.batch_size = 4
        self.batch_size = self.n_rollout_threads // 1
        self.num_sample = args.num_sample

        
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

    def get_intention_loss(self, q_actions, p_actions, q_res, p_res, obs):
        q_action_pred = q_res["action_pred"] # [batch_size * num_vars, num_timesteps, num_inputs]
        # q_action_pred = q_res["action_pred"][:, :-1, :, :] # [batch_size * num_vars, num_timesteps, num_inputs]
        q_target = self.policy.encoder_decoder.decoder.output_feature_norm(check(obs).to(**self.tpdv))
        # q_target = q_actions.ndata['feat'][:, 1:, :]
        q_zIG_rv, q_zIA_rv = q_res["zIG_rv"], q_res["zIA_rv"]
        p_zIG_rv, p_zIA_rv = p_res["zIG_rv"], p_res["zIA_rv"]
        # if self.env_name == "StarCraft2":
        #     predict_rv = torch.distributions.Categorical(logits=q_action_pred)
        #     loss_nll = - predict_rv.log_prob(torch.stack([q_target.squeeze()] * self.num_sample, dim=2)).reshape(-1, self.num_agents, *q_action_pred.shape[1:-1])
        #     b_s = torch.sum(loss_nll, (1, 2))

        # else:
        predict_rv = Normal(q_action_pred, self.policy.sigma_action)
        loss_nll = - predict_rv.log_prob(torch.stack([q_target] * self.num_sample, dim=2)).reshape(-1, *q_action_pred.shape[1:])
        b_s = torch.sum(loss_nll, (1, 3))
        
        # loss_nll = - predict_rv.log_prob(q_target).reshape(-1, self.num_agents, *q_action_pred.shape[1:]) # [batch_size, num_vars, num_timesteps, num_inputs]
        kl_zIG = kl_divergence(q_zIG_rv, p_zIG_rv).reshape(-1, self.num_agents, self.intention_size)  # [batch_size, num_vars, intention_size]
        kl_zIA = kl_divergence(q_zIA_rv, p_zIA_rv).reshape(-1, self.num_agents, self.intention_size)  # [batch_size, num_vars, intention_size]

        # b_s = torch.sum(loss_nll, (1, 2, 3))  # batch_size
        # loss_nll = torch.sum(b_s, dim=1)
        # loss_nll = loss_nll[0]

        # loss_nll = loss_nll.mean() / self.num_agents
        loss_nll = torch.min(b_s, dim=1)[0].mean()
        # loss_nll = b_s.mean() / self.num_agents
        loss_kl = kl_zIG.sum(2).mean() + kl_zIA.sum(2).mean()
        loss = loss_nll + loss_kl

        return {
            "total_loss": loss.mean(),
            "loss_nll": loss_nll.mean(),
            "loss_kl": loss_kl.mean(),
            'kl_zIG': kl_zIG.mean(),
            'kl_zIA': kl_zIA.mean()
        }

    def train_intention(self, past, intention_clip=False, intention_rnn_state_train=None, intention_rnn_state=None, critic_obs = None,
                                            critic_states = None, rnn_states_critic=None, masks=None, correlated_agents=None, update_intention=True, episode=None, imp_weights=None, intentions=None):

        q_actions = check(self.policy.encoder_decoder.build_input(past.reshape(-1, *past.shape[2:])[:, np.newaxis, :,  :])).to(**self.tpdv)
        p_actions = check(self.policy.encoder_decoder.build_input(past.reshape(-1, *past.shape[2:])[:, np.newaxis, :,  :])).to(**self.tpdv) # [batch_size * n_agents, timesteps, input_size]
        obs = check(critic_obs).to(**self.tpdv)
        rnn_states = check(rnn_states_critic[:, 0]).to(**self.tpdv)
        rnn_states_intention = check(intention_rnn_state_train).to(**self.tpdv).reshape(-1, self.recurrent_N, self.intention_hidden_size).transpose(0, 1)
        masks = check(masks).to(**self.tpdv)
        correlated_agents = check(correlated_agents).to(**self.tpdv)
        q_res = self.policy.encoder_decoder.get_intention(q_actions, rnn_states_intention, observ=obs.reshape(-1, *obs.shape[2:])[:, np.newaxis], correlated_agents=correlated_agents.reshape(-1, *correlated_agents.shape[2:])[:, np.newaxis], rnn_states=rnn_states, masks=masks.reshape(-1, *masks.shape[2:])[:, np.newaxis], prior=False, states=critic_states)
        p_res = self.policy.encoder_decoder.get_intention(p_actions, rnn_states_intention, observ=obs.reshape(-1, *obs.shape[2:])[:, np.newaxis], correlated_agents=correlated_agents.reshape(-1, *correlated_agents.shape[2:])[:, np.newaxis], rnn_states=rnn_states, masks=masks.reshape(-1, *masks.shape[2:])[:, np.newaxis])
        loss_dict = self.get_intention_loss(q_actions, p_actions, q_res, p_res, critic_states.reshape(-1, *critic_states.shape[2:])[:, np.newaxis])
        loss = loss_dict["total_loss"]

        # # rebuild loss
        # q_actions = check(self.policy.encoder_decoder.build_input2(past, future)).to(**self.tpdv)
        # res = self.policy.get_intention(q_actions, check(intention_rnn_state).to(**self.tpdv).reshape(-1, self.recurrent_N, self.intention_hidden_size).transpose(0, 1), True)
        # intentions = res["intention"].mean(1).reshape(self.n_rollout_threads, self.episode_length, self.num_agents, self.intention_hidden_size + self.intention_size)
        # # intentions = check(intentions).to(**self.tpdv)
        # rebuild_states_features = self.policy.evaluate_critic(critic_obs, critic_states, masks, rnn_states_critic, intentions, correlated_agents)
        # # rebuilding_loss = mse_loss(check(critic_states).to(**self.tpdv).reshape(-1, critic_states.shape[-1]) - rebuild_states_features).sum()
        # cosine_loss = nn.CosineEmbeddingLoss(margin=0.2, reduction="mean")
        # target = torch.ones(rebuild_states_features.shape[0], device=torch.device(self.device))
        # rebuilding_loss = cosine_loss(check(critic_states).to(**self.tpdv).reshape(-1, critic_states.shape[-1]), rebuild_states_features, target)
        # loss = loss * 0 + rebuilding_loss
        # loss = rebuilding_loss


        

            
            # prev_endecoder_para = self.policy.encoder_decoder.state_dict()

            # if update_intention:
            #     import copy
            #     old_param = copy.deepcopy(self.policy.encoder_decoder).parameters()
            #     self.policy.encoder_decoder_optimizer.zero_grad()
            #     loss.backward()
            #     _ = nn.utils.clip_grad_norm_(self.policy.encoder_decoder.parameters(), self.max_grad_norm)
            #     self.policy.encoder_decoder_optimizer.step()
            #     # self.policy.encoder_decoder_scheduler.step()
                
            #     # soft update open after pre-training
            #     if intention_clip:
            #         for target_param, param in zip(old_param, self.policy.encoder_decoder.parameters()): #   0.0023 0.0018
            #             param.data.copy_(
            #                 target_param.data * (1.0 - self.tau) + param.data * self.tau
            #             )

        return loss, loss_dict["loss_nll"], loss_dict["loss_kl"]  # Normal(batchsize * n_agents, z_dim)

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        if self._use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True, update_critic=True, intention_clip=False, use_intention=False, episode=None, last=False):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, factor_batch, \
        last_actions_batch, intentions_batch, correlated_agents_batch, rnn_states_intention_batch, rnn_states_train_intention_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        factor_batch = check(factor_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        # values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
        #                                                                       obs_batch, 
        #                                                                       rnn_states_batch, 
        #                                                                       rnn_states_critic_batch, 
        #                                                                       actions_batch, 
        #                                                                       masks_batch, 
        #                                                                       available_actions_batch,
        #                                                                       active_masks_batch,
        #                                                                       intentions=intentions_batch,
        #                                                                       use_intention=use_intention, 
        #                                                                       correlated_agents=correlated_agents_batch,
        #                                                                       rnn_states_intention=rnn_states_intention_batch,
        #                                                                       last_actions=np.concatenate([last_actions_past, last_actions_future], axis=1).reshape((-1, *last_actions_past.shape[2:]))
        #                                                                       )
        # # actor update
        # imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        # # imp_weights = torch.prod(torch.exp(action_log_probs - old_action_log_probs_batch),dim=-1,keepdim=True)

        # surr1 = imp_weights * adv_targ
        # surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        # if self._use_policy_active_masks:
        #     policy_action_loss = (-torch.sum(factor_batch * torch.min(surr1, surr2),
        #                                      dim=-1,
        #                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        # else:
        #     policy_action_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # policy_loss = policy_action_loss

        # value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch,
                                                                              use_intention=use_intention, 
                                                                              correlated_agents=correlated_agents_batch,
                                                                              rnn_states_intention=rnn_states_intention_batch,
                                                                              last_actions=last_actions_batch
                                                                              )
                                                                            #   rnn_states_intention=buffer.rnn_states_intention[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *buffer.rnn_states_intention.shape[2:]),                                                                                               
                                                                            #   last_actions=buffer.last_actions.transpose(1, 0, 2, 3).reshape(-1, *buffer.last_actions.shape[2:]),

        self.policy.actor_optimizer.zero_grad()

        # actor update
        # imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        imp_weights = torch.prod(torch.exp(action_log_probs - old_action_log_probs_batch),dim=-1,keepdim=True)


        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(factor_batch * torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()
        self.policy.encoder_decoder_optimizer.zero_grad()

        # if update_actor:             
        #     (policy_loss - dist_entropy * self.entropy_coef).backward()
        ((policy_loss - dist_entropy * self.entropy_coef) + (value_loss * self.value_loss_coef)).backward()

        # actor = list(self.policy.actor.parameters())
        # import copy
        # old_param = list(copy.deepcopy(self.policy.encoder_decoder).parameters())
        # parameters1 = [p for p in self.policy.encoder_decoder.parameters() if p.grad is not None]
        
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)  
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        if update_actor:   
            self.policy.actor_optimizer.step()

        # value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        # self.policy.critic_optimizer.zero_grad()
        

        # if update_critic:  
        #     (value_loss * self.value_loss_coef).backward()
        # parameters = [p for p in self.policy.encoder_decoder.encoder.parameters() if p.grad is not None]
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
            # critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.1)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        if update_critic:  
            self.policy.critic_optimizer.step()

        if self._use_max_grad_norm:
            encoder_decoder_grad_norm = nn.utils.clip_grad_norm_(self.policy.encoder_decoder.parameters(), self.max_grad_norm)
            # critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.1)
        else:
            encoder_decoder_grad_norm = get_gard_norm(self.policy.encoder_decoder.parameters())

        # if update_critic:  
        self.policy.encoder_decoder_optimizer.step()
        
        # intention_loss = self.train_intention(last_actions_past, 
        #                                 last_actions_future, 
        #                                 intention_clip=intention_clip, 
        #                                 intention_rnn_state_train=rnn_states_train_intention_batch, 
        #                                 update_intention=update_intention, 
        #                                 episode=episode
        #                                 ) 

        # self.policy.encoder_decoder_optimizer.zero_grad()

        # if update_intention:                          
        #     (intention_loss).backward()

        # if self._use_max_grad_norm:
        #     _ = nn.utils.clip_grad_norm_(self.policy.encoder_decoder.parameters(), self.max_grad_norm)     
        # else:
        #     _ = get_gard_norm(self.policy.encoder_decoder.parameters())

        # self.policy.encoder_decoder_optimizer.step()
        # self.policy.encoder_decoder_scheduler.step()

        # if update_intention:
        #     # import copy
        #     # old_param = copy.deepcopy(self.policy.encoder_decoder).parameters()
        #     # self.policy.encoder_decoder_optimizer.zero_grad()
        #     loss.backward()
        #     _ = nn.utils.clip_grad_norm_(self.policy.encoder_decoder.parameters(), self.max_grad_norm)
        #     self.policy.encoder_decoder_optimizer.step()
        #     # self.policy.encoder_decoder_scheduler.step()
            
        #     # soft update open after pre-training
        #     if intention_clip:
        #         for target_param, param in zip(old_param, self.policy.encoder_decoder.parameters()): #   0.0023 0.0018
        #             param.data.copy_(
        #                 target_param.data * (1.0 - self.tau) + param.data * self.tau
        #             )

        # ((policy_loss - dist_entropy * self.entropy_coef) + (value_loss * self.value_loss_coef) + intention_loss).backward()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True, intention_clip=False, long_short_clip=False, use_intention=False, episode=None):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        # intention_train_info = self.train_vae(buffer, intention_clip=intention_clip, long_short_clip=long_short_clip, use_intention=use_intention, episode=episode) 
        # train_info.update(intention_train_info)

        # train_info['intention_loss'] = 0
        # train_info['nll_loss'] = 0
        # train_info['kl_loss'] = 0
        # train_info['intention_grad_norm'] = 0

        # if (episode % self.intention_update_freq) == 0:
        #     update_intention = True
        # else:
        #     update_intention = False
        if (episode % self.actor_update_freq) == 0 or episode < self.preupdate:
            update_actor = True
        else:
            update_actor = False
        if (episode % self.critic_update_freq) == 0 or episode < self.preupdate:
            update_critic = True
        else:
            update_critic = False
        
        # self.prep_rollout()

        # values = self.policy.get_values(buffer.share_obs[:-1].transpose(1,0,2).reshape(-1, *buffer.share_obs[:-1].shape[2:]), 
        #                     buffer.rnn_states_critic[:-1].transpose(1, 0, 2, 3).reshape(-1, *buffer.rnn_states_critic.shape[2:]), 
        #                     buffer.masks[:-1].transpose(1,0,2).reshape(-1, * buffer.masks[:-1].shape[2:]), 
        #                     correlated_agents=buffer.correlated_agents[:-1].transpose(1,0,2).reshape(-1, * buffer.correlated_agents[:-1].shape[2:]),  
        #                     rnn_states_intention=buffer.rnn_states_intention[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *buffer.rnn_states_intention.shape[2:]),                                                                  
        #                     last_actions=buffer.last_actions.transpose(1, 0, 2, 3).reshape(-1, *buffer.last_actions.shape[2:]),
        #                     )
    
        # buffer.value_preds[:-1] = values.view(self.episode_length, self.n_rollout_threads, -1).detach().cpu().numpy()

        # next_value = self.policy.get_values(buffer.share_obs[-1], 
        #                                                         buffer.rnn_states_critic[-1],
        #                                                         buffer.masks[-1],
        #                                                         rnn_states_intention=buffer.rnn_states_intention[-1],
        #                                                         last_actions=buffer.last_actions[-1],
        #                                                         correlated_agents=buffer.correlated_agents[-1])
        # next_value = next_value.detach().cpu().numpy()
        # buffer.compute_returns(next_value, self.value_normalizer)

        # self.prep_training()

        # if self._use_popart:
        #     advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        # else:
        #     advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        # advantages_copy = advantages.copy()
        # advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        # mean_advantages = np.nanmean(advantages_copy)
        # std_advantages = np.nanstd(advantages_copy)
        # advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):

            intention_train_info = self.train_vae(buffer, intention_clip=intention_clip, long_short_clip=long_short_clip, use_intention=use_intention, episode=episode) 
            train_info.update(intention_train_info)
            
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length, self.policy.encoder_decoder, long_short_clip, episode)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.ppo_update(sample, 
                                                                                                                                update_actor=update_actor, 
                                                                                                                                intention_clip=intention_clip, 
                                                                                                                                use_intention=use_intention, 
                                                                                                                                episode=episode, 
                                                                                                                                update_critic=update_critic,
                                                                                                                                )

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
            # update_critic = False   
        # self.policy.encoder_decoder_scheduler.step()
        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            if k in ['intention_loss', 'intention_grad_norm', 'nll_loss', 'kl_loss']:
                continue
            train_info[k] /= num_updates
 
        return train_info

    def train_vae(self, buffer, intention_clip=False, long_short_clip=False, use_intention=False, episode=None):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        train_info = {}

        train_info['intention_loss'] = 0
        train_info['nll_loss'] = 0
        train_info['kl_loss'] = 0
        # train_info['rebuilding_loss'] = 0
        train_info['intention_grad_norm'] = 0

        if (episode % self.intention_update_freq) == 0:
            update_intention = True
        else:
            update_intention = False

        train_epoch = 1
        for _ in range(train_epoch):
            intention_data_generator = buffer.intention_recurrent_generator(self.batch_size, self.policy.encoder_decoder, long_short_clip, episode)
            for intention_sample in intention_data_generator:
                last_actions, rnn_states_train_intention_batch, rnn_states_intention_batch, critic_obs_batch, critic_states_batch, rnn_states_critic_batch, masks_batch, intentions, correlated_agents_batch = intention_sample
                intention_loss, nll_loss, kl_loss = self.train_intention(last_actions, 
                                            intention_clip=intention_clip, 
                                            intention_rnn_state_train=rnn_states_train_intention_batch, 
                                            intention_rnn_state=rnn_states_intention_batch, 
                                            critic_obs = critic_obs_batch,
                                            critic_states = critic_states_batch,
                                            rnn_states_critic = rnn_states_critic_batch,
                                            masks = masks_batch,
                                            correlated_agents = correlated_agents_batch,
                                            update_intention=update_intention, 
                                            episode=episode,
                                            intentions=intentions
                                            )
                self.policy.encoder_decoder_optimizer.zero_grad()

                if update_intention:                          
                    (intention_loss).backward()

                # if self._use_max_grad_norm:
                #     intention_grad_norm = nn.utils.clip_grad_norm_(self.policy.encoder_decoder.parameters(), self.max_grad_norm)     
                #     # intention_grad_norm = nn.utils.clip_grad_norm_(self.policy.encoder_decoder.parameters(), 0.1)     
                # else:
                intention_grad_norm = get_gard_norm(self.policy.encoder_decoder.parameters())

                import copy                 
                
                old_param = copy.deepcopy(self.policy.encoder_decoder).parameters()

                self.policy.encoder_decoder_optimizer.step()
                
                # soft update open after pre-training
                if intention_clip:
                    for target_param, param in zip(old_param, self.policy.encoder_decoder.parameters()): #   0.0023 0.0018
                        param.data.copy_(
                            target_param.data * (1.0 - self.tau) + param.data * self.tau
                        )

                train_info['intention_loss'] += intention_loss.item()
                train_info['nll_loss'] += nll_loss.item()
                train_info['kl_loss'] += kl_loss.item()
                # train_info['rebuilding_loss'] += rebuilding_loss.item()
                train_info['intention_grad_norm'] += intention_grad_norm
        train_info['intention_loss'] /= train_epoch * self.n_rollout_threads // self.batch_size
        train_info['nll_loss'] /= train_epoch * self.n_rollout_threads // self.batch_size
        train_info['kl_loss'] /= train_epoch * self.n_rollout_threads // self.batch_size
        # train_info['rebuilding_loss'] /= train_epoch * self.n_rollout_threads // self.batch_size
        train_info['intention_grad_norm'] /=  train_epoch * self.n_rollout_threads // self.batch_size
        # self.policy.encoder_decoder_scheduler.step()
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.encoder_decoder.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.encoder_decoder.eval()
