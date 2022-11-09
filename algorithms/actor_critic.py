import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from utils.util import get_shape_from_obs_space
import numpy as np

def Intention_correlation(intentions, intention_norm, num_agents, intention_size, intention_feature, actor_features_obs, correlated_agents, eval=False):
        # intentions = intentions # 100, 2, 32
        intentions = intentions.detach() # 100, 2, 32
        correlated_agents = correlated_agents.detach() # 100, 2
        # episode_L = int(actor_features_obs.shape[0] / intentions.shape[0])
        # 加入因果推断
        final_intentions = intentions.permute(2, 0, 1) * correlated_agents
        final_intention = final_intentions.permute(1, 2, 0)
        final_intention = final_intention.view(final_intention.shape[0], -1) # 32, 60
        # if eval:
        #     final_intention = final_intention.mean(dim=0).unsqueeze(0)
        # else:
        #     final_intention = final_intention.repeat((episode_L, 1))
        # final_intention = intention_norm(final_intention)
        actor_features = intention_feature(final_intention)
        # actor_features = intention_feature(torch.cat([actor_features_obs, final_intention], dim=-1))
        return actor_features

class BackBone(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, num_agents, train_share_observation_space, device=torch.device("cpu")):
        super(BackBone, self).__init__()
        self.hidden_size = args.hidden_size
        self.intention_hidden_size = args.intention_hidden_size
        self.args=args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_feature_normalization = args.use_feature_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.num_agents = num_agents
        self.intention_size = args.intention_hidden_size + args.intention_size

        obs_shape = get_shape_from_obs_space(obs_space)
        train_share_observation_shape = get_shape_from_obs_space(train_share_observation_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            # self.rnn = RNNLayer(self.hidden_size * 2, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):             
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        
        if self._use_feature_normalization:             
            self.feature_norm = nn.LayerNorm(*train_share_observation_shape)
            # self.feature_norm = nn.LayerNorm(self.hidden_size * 2)
            self.intention_norm = nn.LayerNorm(self.intention_size * num_agents)

        self.intention_feature = nn.Sequential(
                        init_(nn.Linear((self.intention_size * num_agents), self.hidden_size)),
                        nn.ReLU(),        
                        init_(nn.Linear(self.hidden_size, self.hidden_size)),
                    )
        
        self.state_enc = nn.Linear(self.hidden_size * 2, *train_share_observation_shape)
        # self.out_enc = nn.Linear(*train_share_observation_shape, self.hidden_size)

        self.to(device)

    def forward(self, obs, rnn_states, masks, intention=None, use_intention=False, correlated_agents=None, eval=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        intentions = check(intention).to(**self.tpdv)
        correlated_agents = check(correlated_agents).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:             
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if use_intention and intention is not None:            
            intention_features = Intention_correlation(intentions, self.intention_norm, self.num_agents, self.intention_size, self.intention_feature, actor_features, correlated_agents, eval)

        actor_features = torch.cat([actor_features, intention_features], dim=-1)

        # actor_features = self.feature_norm(actor_features)

        actor_features = self.state_enc(actor_features)

        actor_features = self.feature_norm(actor_features)

        # actor_features = self.out_enc(actor_features)

        # actions, action_log_probs, action_logits = self.act(actor_features, available_actions, deterministic)

        return actor_features

class Actor(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, num_agents, train_share_observation_space, base, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.intention_hidden_size = args.intention_hidden_size
        self.args=args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_feature_normalization = args.use_feature_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.num_agents = num_agents
        self.intention_size = args.intention_hidden_size + args.intention_size

        # obs_shape = get_shape_from_obs_space(obs_space)
        train_share_observation_shape = get_shape_from_obs_space(train_share_observation_space)
        # # base = CNNBase if len(obs_shape) == 3 else MLPBase
        # # self.base = base(args, obs_shape)
        # self.base = base

        # if self._use_naive_recurrent_policy or self._use_recurrent_policy:
        #     self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        #     # self.rnn = RNNLayer(self.hidden_size * 2, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):             
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        
        # if self._use_feature_normalization:             
        #     # self.feature_norm = nn.LayerNorm(self.hidden_size)
        #     self.feature_norm = nn.LayerNorm(self.hidden_size * 2)
        #     self.intention_norm = nn.LayerNorm(self.intention_size * num_agents)
        
        # self.state_enc = nn.Linear(self.hidden_size * 2, *train_share_observation_shape)
        self.out_enc = init_(nn.Linear(*train_share_observation_shape, self.hidden_size))

        self.to(device)

    def forward(self, actor_features, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.out_enc(actor_features)

        actions, action_log_probs, action_logits = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs

    def evaluate_actions(self, actor_features, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        if active_masks is not None:             
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.out_enc(actor_features)

        if self.args.algorithm_name=="hatrpo":
            action_log_probs, dist_entropy ,action_mu, action_std, all_probs= self.act.evaluate_actions_trpo(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy


class Critic(nn.Module):
    """
    Critic network class for HAPPO. Outputs value function predictions given centralized input (HAPPO) or local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, num_agents, train_share_observation_space, base, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_feature_normalization = args.use_feature_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        self.intention_size = args.intention_hidden_size + args.intention_size
        self.intention_hidden_size = args.intention_hidden_size
        self.num_agents = num_agents
        self.num_sample = args.num_sample

        # cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        train_share_observation_shape = get_shape_from_obs_space(train_share_observation_space)
        # # base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        # # self.base = base(args, cent_obs_shape)
        # self.base = base

        # if self._use_naive_recurrent_policy or self._use_recurrent_policy:
        #     # self.rnn = RNNLayer(self.hidden_size * 2, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        #     self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # if self._use_feature_normalization:                          
        #     self.feature_norm = nn.LayerNorm(self.hidden_size * 2)
        #     self.intention_norm = nn.LayerNorm(self.intention_size * num_agents)

        # # init_method2 = nn.init.xavier_uniform_
        # # def init2_(m):             
        # #     return init(m, init_method2, lambda x: nn.init.constant_(x, 0))
        # self.intention_feature = nn.Sequential(
        #                 init_(nn.Linear((self.intention_size * num_agents), self.hidden_size)),
        #                 nn.ReLU(),        
        #                 init_(nn.Linear(self.hidden_size, self.hidden_size)),
        #             )

        # self.state_enc = nn.Linear(self.hidden_size * 2, *train_share_observation_shape)
        self.out_enc = init_(nn.Linear(*train_share_observation_shape, self.hidden_size)) 

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, critic_features):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        critic_features = self.out_enc(critic_features)

        values = self.v_out(critic_features)

        return values
    
    def evaluate_critic(self, obs, cent_obs, masks, rnn_states=None, intentions=None, correlated_agents=None):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv).reshape(-1, obs.shape[-1])
        rnn_states = check(rnn_states).to(**self.tpdv).reshape(-1, self._recurrent_N, rnn_states.shape[-1])
        masks = check(masks).to(**self.tpdv).reshape(-1, masks.shape[-1])
        intentions = check(intentions).to(**self.tpdv).reshape(-1, self.num_agents, intentions.shape[-1])
        correlated_agents = check(correlated_agents).to(**self.tpdv).reshape(-1, correlated_agents.shape[-1])

        rebuild_states_features = self.base(obs, rnn_states, masks, intentions, use_intention=True, correlated_agents=correlated_agents, eval=False)
        # critic_states_features = self.base(cent_obs)

        # if self._use_naive_recurrent_policy or self._use_recurrent_policy:
        #     critic_obs_features, rnn_states = self.rnn(critic_obs_features, rnn_states, masks)
        #     # critic_states_features, rnn_states = self.rnn(critic_states_features, rnn_states, masks)
        
        # intentions_feature = self.intention_feature(intentions)
        # rebuild_states_features = torch.cat([critic_obs_features, intentions_feature], dim=-1)
        # rebuild_states_features = self.feature_norm(rebuild_states_features)

        # rebuild_states_features = self.state_enc(rebuild_states_features)

        # rebuild_states_features = self.out_enc(rebuild_states_features)

        return rebuild_states_features
