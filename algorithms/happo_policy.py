from multiprocessing.context import ForkContext
import torch
from algorithms.actor_critic import Actor, Critic, BackBone
from utils.util import update_linear_schedule, get_shape_from_obs_space
from algorithms.encoder_decoder import Encoder_Decoder
from torch import nn 
import torch.nn.functional as F 
from algorithms.utils.util import init, check 
from torch.distributions import Normal, kl_divergence
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
import numpy as np

class HAPPO_Policy:
    """
    HAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for HAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, train_share_observation_space, device=torch.device("cpu")):
        self.args=args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.hidden_size = args.hidden_size
        self.intention_hidden_size = args.intention_hidden_size
        self.intention_size = get_shape_from_obs_space(train_share_observation_space)[0]
        self.act_space = act_space
        self._recurrent_N = args.recurrent_N
        self.n_rollout_threads = self.args.n_rollout_threads
        self.vae_lr = args.vae_lr
        self.num_agents = num_agents
        self.num_sample = args.num_sample
        self.tpdv = dict(dtype=torch.float32, device=device)

        # self.backbone = BackBone(args, self.obs_space, self.act_space, self.num_agents, train_share_observation_space, self.device)
        self.encoder_decoder = Encoder_Decoder(args, self.act_space, self.num_agents, train_share_observation_space, self.obs_space, self.device) 

        self.actor = Actor(args, self.obs_space, self.act_space, self.num_agents, train_share_observation_space, self.encoder_decoder, self.device)

        ######################################Please Note#########################################
        #####   We create one critic for each agent, but they are trained with same data     #####
        #####   and using same update setting. Therefore they have the same parameter,       #####
        #####   you can regard them as the same critic.                                      #####
        ##########################################################################################
        self.critic = Critic(args, self.share_obs_space, self.num_agents, train_share_observation_space, self.encoder_decoder, self.device)


        # self.sigma_action = torch.tensor([1e-3]).to(self.device)
        self.sigma_action = nn.Parameter(torch.tensor([1e-1]).clamp(min=1e-3)).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

        self.encoder_decoder_optimizer = torch.optim.Adam(self.encoder_decoder.parameters(), lr=self.vae_lr, weight_decay=self.weight_decay) 
        # self.encoder_decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_decoder_optimizer, T_max=100, eta_min=0, last_epoch=-1, verbose=False)  
        # self.encoder_decoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.encoder_decoder_optimizer, gamma=0.9)
        # self.encoder_decoder_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.encoder_decoder_optimizer,max_lr=self.vae_lr,pct_start=0.3,total_steps=2500,div_factor=100,final_div_factor=10000) 
        # self.critic_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.critic_optimizer,max_lr=self.critic_lr,pct_start=0.2,total_steps=2500,div_factor=100,final_div_factor=1) 

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        # if episode == 125:
        #     critic_lr = 5e-3
        #     for param_group in self.critic_optimizer.param_groups:
        #         param_group['lr'] = critic_lr
        #     lr = 5e-4
        #     for param_group in self.actor_optimizer.param_groups:
        #         param_group['lr'] = lr
            # update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
            # update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        if episode <=250:
            # update_linear_schedule(self.encoder_decoder_optimizer, episode, episodes, self.vae_lr)
            self.critic_scheduler.step()

    def get_actions(self, cent_obs, obs, rnn_states, masks, correlated_agents=None, rnn_states_intention=None, last_actions=None,
                    available_actions=None, deterministic=False, use_intention=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        graph = self.encoder_decoder.build_input(np.array(last_actions[:, :, np.newaxis, :]).transpose(0, 2, 1, 3)) 
        obs = check(obs[:, np.newaxis]).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states_intention = check(rnn_states_intention).to(**self.tpdv)
        masks = check(masks[:, np.newaxis]).to(**self.tpdv)
        correlated_agents = check(correlated_agents[:, np.newaxis]).to(**self.tpdv)

        rnn_states_intention = rnn_states_intention.reshape(-1, self._recurrent_N, self.intention_hidden_size).transpose(0, 1) # batchsize * n_agents, 1, 64 -> 1, batchsize * n_agents, 64
        q_res = self.encoder_decoder.get_intention(graph, rnn_states_intention, observ=obs, correlated_agents=correlated_agents, rnn_states=rnn_states, masks=masks, forecast=True)
        rnn_states_intention = q_res["temp_rnn_state"].transpose(0, 1).reshape(-1, self.num_agents, self._recurrent_N, self.intention_hidden_size) # (1, batchsize * n_agents, hidden_size) - > (batchsize * n_agents, 1, hidden_size)
        rnn_states = q_res["rnn_states"].mean(1) # (1, batchsize * n_agents, hidden_size) - > (batchsize * n_agents, 1, hidden_size)
        obs_features = q_res["intention"].mean(1)

        actions, action_log_probs = self.actor(obs_features, available_actions, deterministic)

        values = self.critic(obs_features)

        return values, actions, action_log_probs, rnn_states, rnn_states_intention

    def get_values(self, obs, rnn_states, masks, correlated_agents=None, rnn_states_intention=None, last_actions=None):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        graph = self.encoder_decoder.build_input(np.array(last_actions[:, :, np.newaxis, :]).transpose(0, 2, 1, 3)) 
        obs = check(obs[:, np.newaxis]).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states_intention = check(rnn_states_intention).to(**self.tpdv)
        masks = check(masks[:, np.newaxis]).to(**self.tpdv)
        correlated_agents = check(correlated_agents[:, np.newaxis]).to(**self.tpdv)

        rnn_states_intention = rnn_states_intention.reshape(-1, self._recurrent_N, self.intention_hidden_size).transpose(0, 1) # batchsize * n_agents, 1, 64 -> 1, batchsize * n_agents, 64
        q_res = self.encoder_decoder.get_intention(graph, rnn_states_intention, observ=obs, correlated_agents=correlated_agents, rnn_states=rnn_states, masks=masks, forecast=True)
        rnn_states_intention = q_res["temp_rnn_state"].transpose(0, 1).reshape(-1, self.num_agents, self._recurrent_N, self.intention_hidden_size) # (1, batchsize * n_agents, hidden_size) - > (batchsize * n_agents, 1, hidden_size)
        rnn_states = q_res["rnn_states"].mean(1) # (1, batchsize * n_agents, hidden_size) - > (batchsize * n_agents, 1, hidden_size)
        obs_features = q_res["intention"].mean(1)

        values = self.critic(obs_features)
        return values

    def evaluate_actions(self, obs, rnn_states, action, masks,
                         available_actions=None, active_masks=None, use_intention=False, correlated_agents=None, rnn_states_intention=None, last_actions=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        graph = self.encoder_decoder.build_input(np.array(last_actions[:, :, np.newaxis, :]).transpose(0, 2, 1, 3)) 
        obs = check(obs[:, np.newaxis]).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states_intention = check(rnn_states_intention).to(**self.tpdv)
        masks = check(masks[:, np.newaxis]).to(**self.tpdv)
        correlated_agents = check(correlated_agents[:, np.newaxis]).to(**self.tpdv)

        rnn_states_intention = rnn_states_intention.reshape(-1, self._recurrent_N, self.intention_hidden_size).transpose(0, 1) # batchsize * n_agents, 1, 64 -> 1, batchsize * n_agents, 64
        q_res = self.encoder_decoder.get_intention(graph, rnn_states_intention, observ=obs, correlated_agents=correlated_agents, rnn_states=rnn_states, masks=masks, forecast=True)
        rnn_states_intention = q_res["temp_rnn_state"].transpose(0, 1).reshape(-1, self.num_agents, self._recurrent_N, self.intention_hidden_size) # (1, batchsize * n_agents, hidden_size) - > (batchsize * n_agents, 1, hidden_size)
        rnn_states = q_res["rnn_states"].mean(1) # (1, batchsize * n_agents, hidden_size) - > (batchsize * n_agents, 1, hidden_size)
        obs_features = q_res["intention"].mean(1)

        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs_features,
                                                                action,
                                                                available_actions,
                                                                active_masks, 
                                                                )

        values = self.critic(obs_features)

        return values, action_log_probs, dist_entropy

    def evaluate_critic(self, obs, cent_obs, masks, rnn_states, intentions, correlated_agents):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        rebuild_states_features = self.critic.evaluate_critic(obs, cent_obs, masks, rnn_states, intentions, correlated_agents)

        return rebuild_states_features


    def act(self, obs, rnn_states, masks, available_actions=None, deterministic=False, correlated_agents=None, rnn_states_intention=None, last_actions=None, eval=True):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        graph = self.encoder_decoder.build_input(np.array(last_actions[:, :, np.newaxis, :]).transpose(0, 2, 1, 3)) 
        obs = check(obs[:, np.newaxis]).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states_intention = check(rnn_states_intention).to(**self.tpdv)
        masks = check(masks[:, np.newaxis]).to(**self.tpdv)         
        correlated_agents = check(correlated_agents[:, np.newaxis]).to(**self.tpdv)

        rnn_states_intention = rnn_states_intention.reshape(-1, self._recurrent_N, self.intention_hidden_size).transpose(0, 1) # batchsize * n_agents, 1, 64 -> 1, batchsize * n_agents, 64
        q_res = self.encoder_decoder.get_intention(graph, rnn_states_intention, observ=obs, correlated_agents=correlated_agents, rnn_states=rnn_states, masks=masks, forecast=True)
        rnn_states_intention = q_res["temp_rnn_state"].transpose(0, 1).reshape(-1, self.num_agents, self._recurrent_N, self.intention_hidden_size) # (1, batchsize * n_agents, hidden_size) - > (batchsize * n_agents, 1, hidden_size)
        rnn_states = q_res["rnn_states"].mean(1) # (1, batchsize * n_agents, hidden_size) - > (batchsize * n_agents, 1, hidden_size)
        obs_features = q_res["intention"].mean(1)

        actions, _ = self.actor(obs_features, available_actions, deterministic)

        return actions, rnn_states, rnn_states_intention

    def get_intention(self, graph, intention_rnn_state_train=None, observ=None, correlated_agents=None, rnn_states=None, masks=None, forecast=False, clip=False):
        actions = graph.ndata['feat']
        zIG_rv, zIA_rv, temp_rnn_state = self.encoder_decoder.encoder(graph, actions, hidden=intention_rnn_state_train)
        if clip:
            return temp_rnn_state
        # if forecast == True:
        #     zIA = torch.stack([zIA_rv.mean()] * self.num_sample, dim=0).transpose(1, 0)
        #     zIG = torch.stack([zIG_rv.mean()] * self.num_sample, dim=0).transpose(1, 0)
        # else:
        zIA = zIA_rv.rsample([self.num_sample]).transpose(1, 0)
        zIG = zIG_rv.rsample([self.num_sample]).transpose(1, 0)

        graph.ndata['zIA'] = zIA
        graph.ndata['zIG'] = zIG
        if forecast:
            intention, rnn_states = self.encoder_decoder.decoder(graph, observ, correlated_agents=correlated_agents, rnn_states=rnn_states, masks=masks, forecast=True)
            # intention = self.encoder_decoder.decoder(graph, graph.ndata['feat'], forecast=True)

            res = {
                "intention": intention,
                "zIG_rv": zIG_rv,  # Normal(batchsize * n_agents, z_dim)
                "zIA_rv": zIA_rv,  # Normal(batchsize * n_agents, z_dim)
                "zIG": zIG,
                "zIA": zIA,
                "temp_rnn_state": temp_rnn_state,
                "critic_rnn_state": rnn_states
            }
        else:
            action_pred = self.encoder_decoder.decoder(graph, graph.ndata['feat'])

            res = {
                "action_pred": action_pred,
                "zIG_rv": zIG_rv,  # Normal(batchsize * n_agents, z_dim)
                "zIA_rv": zIA_rv,  # Normal(batchsize * n_agents, z_dim)
                "zIG": zIG,
                "zIA": zIA,
                "temp_rnn_state": temp_rnn_state
            }

        return res
    
    def Correlated_Agents(self, intention, available_actions, actor_features_obs, rnn_states, masks):
        intentions = check(intention).to(**self.tpdv)
        episode_L = int(actor_features_obs.shape[0] / intentions.shape[0])
        # 加入因果推断或KL散度
        
        correlated_agents = np.zeros((intentions.shape[0], self.num_agents), dtype=np.float32)
        kl_probs = np.zeros((intentions.shape[0], self.num_agents), dtype=np.float32)
        kl_coef = np.zeros((intentions.shape[0]), dtype=np.float32)
        if self.args.causal_inference_or_kl: # CI
            with torch.no_grad():
                # for env_idx in range(intentions.shape[0]):
                for agent_idx in range(intentions.shape[1]):
                    another_intention = intentions.clone() # 2, 32
                    # another_intention = another_intention.reshape(1, *another_intention.shape)
                    another_intention[:, agent_idx] = torch.zeros([intentions.shape[0], intentions.shape[-1]]) # 1, 2, 32
                    another_intention = another_intention.view(intentions.shape[0], -1)
                    another_intention = self.actor.intention_norm(another_intention)
                    another_intention_features = self.actor.intention_feature(another_intention)
                    another_actor_features = torch.cat([actor_features_obs.reshape(intentions.shape[0], actor_features_obs.shape[-1]), another_intention_features], dim=-1)
                    another_actor_features = self.actor.feature_norm(another_actor_features)
                
                    temp_intention = intentions.clone()
                    # temp_intention = temp_intention.reshape(intentions.shape[0], *temp_intention.shape)
                    temp_intention = temp_intention.view(intentions.shape[0], -1) # 32, 60
                    temp_intention = self.actor.intention_norm(temp_intention)
                    temp_intention_features = self.actor.intention_feature(temp_intention)                     
                    temp_actor_features = torch.cat([actor_features_obs.reshape(intentions.shape[0], actor_features_obs.shape[-1]), temp_intention_features], dim=-1)                 
                    temp_actor_features = self.actor.feature_norm(temp_actor_features)
                    # temp_actor_features = self.actor.intention_feature(torch.cat([actor_features_obs[env_idx].reshape(1, actor_features_obs.shape[-1]), temp_intention.repeat((episode_L, 1))], dim=-1))

                    if self.args.use_naive_recurrent_policy or self.args.use_recurrent_policy:
                        another_actor_features, _ = self.actor.rnn(another_actor_features, rnn_states.reshape(intentions.shape[0], *rnn_states.shape[1:]), masks.reshape(intentions.shape[0], masks.shape[-1]))
                        temp_actor_features, _ = self.actor.rnn(temp_actor_features, rnn_states.reshape(intentions.shape[0], *rnn_states.shape[1:]), masks.reshape(intentions.shape[0], masks.shape[-1]))

                    _, _, another_action_logits = self.actor.act(another_actor_features, available_actions, False)
                    _, _, temp_action_logits = self.actor.act(temp_actor_features, available_actions, False)

                    kl_prob = kl_divergence(another_action_logits, temp_action_logits)
                        # KLDivLoss = nn.KLDivLoss(reduction='batchmean')  
                        # if isinstance(another_action_logits, Normal):
                        #     kl_prob = kl_divergence(another_action_logits, temp_action_logits)
                        #     # kl_prob = self.actor.act.kl_divergence(temp_action_logits.mean, another_action_logits.mean, temp_action_logits.stddev, another_action_logits.stddev).mean()       
                        #     # p_output = F.softmax(temp_action_logits.mean, dim=-1)                                  
                        #     # q_output = F.softmax(another_action_logits.mean, dim=-1)                              
                        #     # log_mean_output = ((p_output + q_output )/2).log()                              
                        #     # kl_prob = (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2
                        # else:                             
                        #     p_output = F.softmax(temp_action_logits.logits, dim=-1)                                  
                        #     q_output = F.softmax(another_action_logits.logits, dim=-1)                              
                        #     log_mean_output = ((p_output + q_output )/2).log()                              
                        #     kl_prob = (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

                    kl_prob = kl_prob.mean(dim=-1)
                    kl_probs[:, agent_idx] = kl_prob.cpu().numpy()
                kl_coef = np.percentile(kl_probs, 20, axis=-1, keepdims=True)
                correlated_agents = np.where(kl_probs >= kl_coef, 1., 0.)

        return correlated_agents

    # def Correlated_Agents(self, intention, available_actions, actor_features_obs, rnn_states, masks):
    #     intentions = check(intention).to(**self.tpdv)
    #     episode_L = int(actor_features_obs.shape[0] / intentions.shape[0])
    #     # 加入因果推断或KL散度
        
    #     correlated_agents = np.zeros((intentions.shape[0], self.num_agents), dtype=np.float32)
    #     kl_probs = np.zeros((intentions.shape[0], self.num_agents), dtype=np.float32)
    #     kl_coef = np.zeros((intentions.shape[0]), dtype=np.float32)
    #     if self.args.causal_inference_or_kl: # CI
    #         with torch.no_grad():
    #             for env_idx in range(intentions.shape[0]):
    #                 for agent_idx in range(intentions.shape[1]):
    #                     another_intention = intentions[env_idx].clone() # 2, 32
    #                     another_intention = another_intention.reshape(1, *another_intention.shape)
    #                     another_intention[:, agent_idx] = torch.zeros([1, intentions.shape[-1]]) # 1, 2, 32
    #                     another_intention = another_intention.view(1, -1)
    #                     another_intention = self.actor.intention_norm(another_intention)
    #                     another_intention_features = self.actor.intention_feature(another_intention)
    #                     another_actor_features = torch.cat([actor_features_obs[env_idx].reshape(1, actor_features_obs.shape[-1]), another_intention_features], dim=-1)
    #                     another_actor_features = self.actor.feature_norm(another_actor_features)
                    
    #                     temp_intention = intentions[env_idx].clone()
    #                     temp_intention = temp_intention.reshape(1, *temp_intention.shape)
    #                     temp_intention = temp_intention.view(1, -1) # 32, 60
    #                     temp_intention = self.actor.intention_norm(temp_intention)
    #                     temp_intention_features = self.actor.intention_feature(temp_intention)                     
    #                     temp_actor_features = torch.cat([actor_features_obs[env_idx].reshape(1, actor_features_obs.shape[-1]), temp_intention_features], dim=-1)                 
    #                     temp_actor_features = self.actor.feature_norm(temp_actor_features)
    #                     # temp_actor_features = self.actor.intention_feature(torch.cat([actor_features_obs[env_idx].reshape(1, actor_features_obs.shape[-1]), temp_intention.repeat((episode_L, 1))], dim=-1))

    #                     if self.args.use_naive_recurrent_policy or self.args.use_recurrent_policy:
    #                         another_actor_features, _ = self.actor.rnn(another_actor_features, rnn_states[env_idx].reshape(1, *rnn_states.shape[1:]), masks[env_idx].reshape(1, masks.shape[-1]))
    #                         temp_actor_features, _ = self.actor.rnn(temp_actor_features, rnn_states[env_idx].reshape(1, *rnn_states.shape[1:]), masks[env_idx].reshape(1, masks.shape[-1]))

    #                     _, _, another_action_logits = self.actor.act(another_actor_features, available_actions, False)
    #                     _, _, temp_action_logits = self.actor.act(temp_actor_features, available_actions, False)

    #                     kl_prob = kl_divergence(another_action_logits, temp_action_logits)
    #                         # KLDivLoss = nn.KLDivLoss(reduction='batchmean')  
    #                         # if isinstance(another_action_logits, Normal):
    #                         #     kl_prob = kl_divergence(another_action_logits, temp_action_logits)
    #                         #     # kl_prob = self.actor.act.kl_divergence(temp_action_logits.mean, another_action_logits.mean, temp_action_logits.stddev, another_action_logits.stddev).mean()       
    #                         #     # p_output = F.softmax(temp_action_logits.mean, dim=-1)                                  
    #                         #     # q_output = F.softmax(another_action_logits.mean, dim=-1)                              
    #                         #     # log_mean_output = ((p_output + q_output )/2).log()                              
    #                         #     # kl_prob = (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2
    #                         # else:                             
    #                         #     p_output = F.softmax(temp_action_logits.logits, dim=-1)                                  
    #                         #     q_output = F.softmax(another_action_logits.logits, dim=-1)                              
    #                         #     log_mean_output = ((p_output + q_output )/2).log()                              
    #                         #     kl_prob = (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

    #                     kl_prob = kl_prob.mean().item()
    #                     kl_probs[env_idx, agent_idx] = kl_prob
    #                 kl_coef[env_idx] = np.percentile(kl_probs[env_idx], 40)
    #                 correlated_agents[env_idx] = np.where(kl_probs[env_idx] >= kl_coef[env_idx], 1., 0.)

    #     return correlated_agents