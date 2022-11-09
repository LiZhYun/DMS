import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer2, RNNLayer
from algorithms.utils.act import ACTLayer
from utils.util import get_shape_from_act_space, get_shape_from_obs_space
import numpy as np
from torch.distributions import Normal
from dgl.nn.pytorch.conv import GraphConv
import dgl

class GraphAttentionConv(nn.Module):
    def __init__(self, h_dim, z_dim, attn_head, K=1):
        super(GraphAttentionConv, self).__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.K = K
        self.attn_head = attn_head
        self.attn_s = nn.ModuleList()
        self.attn_d = nn.ModuleList()
        for k in range(K):
            self.attn_s.append(nn.Linear(self.z_dim, h_dim, bias=False))
            self.attn_d.append(nn.Linear(self.z_dim, h_dim, bias=False))

        self.F1 = nn.Sequential(nn.Linear(self.h_dim * self.K, self.h_dim, bias=False),
                                nn.ReLU(),
                                nn.Linear(self.h_dim, self.h_dim, bias=False))
        self.F2 = nn.Sequential(nn.Linear(2 * h_dim, self.h_dim, bias=False),
                                nn.ReLU(),
                                nn.Linear(self.h_dim, self.h_dim, bias=False))

    def edge_attention(self, edges):
        a = []
        for k in range(self.K):
            a_s = self.attn_s[k](edges.src['zIG'])
            a_d = self.attn_d[k](edges.dst['zIG'])
            a.append(F.leaky_relu((a_s * a_d).sum(-1).unsqueeze(-1)))

        return {'eIG': torch.cat(a, dim=-1)}

    def message_func(self, edges):
        dA = self.F2(
            torch.cat([edges.src['xt_enc'], edges.dst['xt_enc']], dim=-1))

        return {'dA': dA, 'eIG': edges.data['eIG']}

    def reduce_func(self, nodes):

        # calculate attention weight
        alpha = F.softmax(nodes.mailbox['eIG'], dim=1)
        res = []
        for k in range(self.K):
            res.append(torch.mean(alpha[:, :, :, k].unsqueeze(
                -1) * nodes.mailbox['dA'], dim=1))
        deltax = self.F1(torch.cat(res, dim=-1))
        return {'deltax': deltax, "alpha": alpha}

    def attn(self, graph):
        graph.apply_edges(self.edge_attention)
        return graph

    def conv(self, graph):
        graph.update_all(self.message_func, self.reduce_func)
        return graph

class Encoder_Decoder(nn.Module):
    def __init__(self, args, act_space, num_agents, train_share_observation_space, obs_space, device=torch.device("cpu")):
        super(Encoder_Decoder, self).__init__()
        self.args = args
        self.num_sample = args.num_sample
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = num_agents
        self.encoder = Encoder(args, act_space, num_agents, train_share_observation_space, device)
        self.prior = Prior(args, act_space, num_agents, device)
        self.decoder = Decoder(args, act_space, train_share_observation_space, obs_space, num_agents, device)
    
    def build_input(self, past, future=None):
        # observ [bz, obs_shape]
        if future is not None:
            actions = np.concatenate([past, future], axis=1)
        else:
            actions = past

        num_agents = self.num_agents
        off_diag = np.ones([num_agents, num_agents]) - np.eye(num_agents)
        rel_src = np.where(off_diag)[0]
        rel_dst = np.where(off_diag)[1]

        actions = check(actions.transpose(0, 2, 1, 3)).to(**self.tpdv)  # [bz, n_agents, timesteps, feat]

        N = actions.shape[0]
        graphs = []
        for ii in range(N):
            graph = dgl.graph((rel_src, rel_dst)).to(self.device)
            graph.ndata["feat"] = actions[ii]
            graphs.append(graph)
        # actions = actions.reshape((-1, *actions.shape[2:])) # [batch_size * n_agents, timesteps, input_size]
        graphs = dgl.batch(graphs)
        return graphs

    def build_input2(self, past, future=None):
        if future is not None:
            actions = np.concatenate([past, future], axis=1)
        else:
            actions = past

        num_agents = self.num_agents
        off_diag = np.ones([num_agents, num_agents]) - np.eye(num_agents)
        rel_src = np.where(off_diag)[0]
        rel_dst = np.where(off_diag)[1]

        actions = check(actions.reshape(-1, self.num_agents, 1, actions.shape[-1])).to(**self.tpdv)  # [bz, n_agents, timesteps, feat]

        N = actions.shape[0]
        graphs = []
        for ii in range(N):
            graph = dgl.graph((rel_src, rel_dst)).to(self.device)
            graph.ndata["feat"] = actions[ii]
            graphs.append(graph)
        # actions = actions.reshape((-1, *actions.shape[2:])) # [batch_size * n_agents, timesteps, input_size]
        graphs = dgl.batch(graphs)
        return graphs
    
    def get_intention(self, graph, intention_rnn_state_train=None, observ=None, correlated_agents=None, rnn_states=None, masks=None, forecast=False, clip=False, prior=True, states=None):
        actions = graph.ndata['feat']
        if prior:
            zIG_rv, zIA_rv, temp_rnn_state = self.prior(graph, actions, hidden=intention_rnn_state_train)
        else:
            states = self.decoder.output_feature_norm(check(states).to(**self.tpdv))
            zIG_rv, zIA_rv, temp_rnn_state = self.encoder(graph, actions, hidden=intention_rnn_state_train, states=states)

        if clip:
            return temp_rnn_state
        # if forecast == True:
        #     zIA = torch.stack([zIA_rv.mean] * self.num_sample, dim=0).transpose(1, 0)
        #     zIG = torch.stack([zIG_rv.mean] * self.num_sample, dim=0).transpose(1, 0)
        # else:
        zIA = zIA_rv.rsample([self.num_sample]).transpose(1, 0)
        zIG = zIG_rv.rsample([self.num_sample]).transpose(1, 0)
        graph.ndata['zIA'] = zIA
        graph.ndata['zIG'] = zIG
        if forecast:
            intention, rnn_states = self.decoder(graph, graph.ndata['feat'], observ, correlated_agents=correlated_agents, rnn_states=rnn_states, masks=masks, forecast=True)
            # intention = self.encoder_decoder.decoder(graph, graph.ndata['feat'], forecast=True)

            res = {
                "intention": intention,
                "zIG_rv": zIG_rv,  # Normal(batchsize * n_agents, z_dim)
                "zIA_rv": zIA_rv,  # Normal(batchsize * n_agents, z_dim)
                "zIG": zIG,
                "zIA": zIA,
                "temp_rnn_state": temp_rnn_state,
                "rnn_states": rnn_states
            }
        else:
            action_pred = self.decoder(graph, graph.ndata['feat'], observ, correlated_agents=correlated_agents, rnn_states=rnn_states, masks=masks, forecast=False)

            res = {
                "action_pred": action_pred,
                "zIG_rv": zIG_rv,  # Normal(batchsize * n_agents, z_dim)
                "zIA_rv": zIA_rv,  # Normal(batchsize * n_agents, z_dim)
                "zIG": zIG,
                "zIA": zIA,
                "temp_rnn_state": temp_rnn_state
            }

        return res

class Encoder(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, action_space, num_agents, train_share_observation_space, device=torch.device("cpu")):
        super(Encoder, self).__init__()
        self.hidden_size = args.intention_hidden_size
        self.intention_size = args.intention_size
        self.args=args
        self.num_agents = num_agents
        self._gain = nn.init.calculate_gain('relu')
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks  # by default True, whether to mask useless data in policy loss
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy # by default False, use the whole trajectory to calculate hidden states.
        self._use_recurrent_policy = args.use_recurrent_policy # by default, use Recurrent Policy. If set, do not use.
        self._recurrent_N = args.recurrent_N
        self._use_feature_normalization = args.use_feature_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)
        rnn_input_dim = get_shape_from_act_space(action_space)
        self.rnn = RNNLayer2(rnn_input_dim, self.hidden_size, self._recurrent_N, self._use_orthogonal, self.num_agents)
        self.gcn = GraphConv(self.hidden_size, self.hidden_size)
        self.x_enc = MLPBase(args, get_shape_from_obs_space(train_share_observation_space))
        self.x2_enc = nn.Linear(self.hidden_size * 2, self.hidden_size)
                                    
        # for name, param in self.rnn.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0)
        #     elif 'weight' in name:
        #         if self._use_orthogonal:
        #             nn.init.orthogonal_(param)
        #         else:
        #             nn.init.xavier_uniform_(param)
        
        if self._use_feature_normalization:
            self.input_feature_norm = nn.LayerNorm(rnn_input_dim)
            self.feature_norm = nn.LayerNorm(self.hidden_size)
        # self.dropout = nn.Dropout(p=0.5)
        self.zIA_mu_enc = nn.Linear(self.hidden_size, self.intention_size)
        self.zIA_std_enc = nn.Sequential(
            nn.Linear(self.hidden_size, self.intention_size),
            nn.Softplus())
        self.zIG_mu_enc = nn.Linear(self.hidden_size, self.intention_size)
        self.zIG_std_enc = nn.Sequential(
            nn.Linear(self.hidden_size, self.intention_size),
            nn.Softplus())
        self.init_weights()
        self.to(device)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, graph, actions, states, hidden=None):
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
        # actions shape: [batchsize * n_agents, num_timesteps, num_dims] -> [num_timesteps, batchsize * n_agents, num_dims]
        actions = check(actions).to(**self.tpdv)
        states = check(states.reshape(-1, *states.shape[2:])).to(**self.tpdv).repeat(self.num_agents, 1)
        actions = actions.transpose(1, 0)

        # if self._use_feature_normalization:
        #     x = self.input_feature_norm(actions)
        if hidden is None:
            _, temp_rnn_state = self.rnn(actions, hidden)  # (1, batchsize * n_agents, hidden_size)
        else:
            hidden = check(hidden).to(**self.tpdv)
            _, temp_rnn_state = self.rnn(actions, hidden)
        x = temp_rnn_state.squeeze(0)
        x = self.feature_norm(x)
        x = self.gcn(graph, x)
        x = torch.cat([x, self.x_enc(states)], dim=-1)    
        x = self.x2_enc(x)  
        # x = x + self.x_enc(states)
        # x = self.dropout(x)
        zIA_mu = self.zIA_mu_enc(x)  # Encoder Eq. 5 # (batchsize * num_atoms, z_dim)
        zIA_std = self.zIA_std_enc(x)
        zIG_mu = self.zIG_mu_enc(x)  # Encoder Eq. 5 # (batchsize * num_atoms, z_dim)
        zIG_std = self.zIG_std_enc(x)
        
        return Normal(zIG_mu, zIG_std), Normal(zIA_mu, zIA_std), temp_rnn_state

class Prior(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, action_space, num_agents, device=torch.device("cpu")):
        super(Prior, self).__init__()
        self.hidden_size = args.intention_hidden_size
        self.intention_size = args.intention_size
        self.args=args
        self.num_agents = num_agents
        self._gain = nn.init.calculate_gain('relu')
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks  # by default True, whether to mask useless data in policy loss
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy # by default False, use the whole trajectory to calculate hidden states.
        self._use_recurrent_policy = args.use_recurrent_policy # by default, use Recurrent Policy. If set, do not use.
        self._recurrent_N = args.recurrent_N
        self._use_feature_normalization = args.use_feature_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)
        rnn_input_dim = get_shape_from_act_space(action_space)
        self.rnn = RNNLayer2(rnn_input_dim, self.hidden_size, self._recurrent_N, self._use_orthogonal, self.num_agents)
        self.gcn = GraphConv(self.hidden_size, self.hidden_size)
        
        # for name, param in self.rnn.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0)
        #     elif 'weight' in name:
        #         if self._use_orthogonal:
        #             nn.init.orthogonal_(param)
        #         else:
        #             nn.init.xavier_uniform_(param)
        
        if self._use_feature_normalization:
            self.input_feature_norm = nn.LayerNorm(rnn_input_dim)
            self.feature_norm = nn.LayerNorm(self.hidden_size)
        # self.dropout = nn.Dropout(p=0.5)
        self.zIA_mu_enc = nn.Linear(self.hidden_size, self.intention_size)
        self.zIA_std_enc = nn.Sequential(
            nn.Linear(self.hidden_size, self.intention_size),
            nn.Softplus())
        self.zIG_mu_enc = nn.Linear(self.hidden_size, self.intention_size)
        self.zIG_std_enc = nn.Sequential(
            nn.Linear(self.hidden_size, self.intention_size),
            nn.Softplus())
        self.init_weights()
        self.to(device)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, graph, actions, hidden=None):
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
        # actions shape: [batchsize * n_agents, num_timesteps, num_dims] -> [num_timesteps, batchsize * n_agents, num_dims]
        actions = check(actions).to(**self.tpdv)
        actions = actions.transpose(1, 0)

        # if self._use_feature_normalization:
        #     x = self.input_feature_norm(actions)
        if hidden is None:
            _, temp_rnn_state = self.rnn(actions, hidden)  # (1, batchsize * n_agents, hidden_size)
        else:
            hidden = check(hidden).to(**self.tpdv)
            _, temp_rnn_state = self.rnn(actions, hidden)
        x = temp_rnn_state.squeeze(0)
        x = self.feature_norm(x)
        x = self.gcn(graph, x)
        # x = self.dropout(x)
        zIA_mu = self.zIA_mu_enc(x)  # Encoder Eq. 5 # (batchsize * num_atoms, z_dim)
        zIA_std = self.zIA_std_enc(x)
        zIG_mu = self.zIG_mu_enc(x)  # Encoder Eq. 5 # (batchsize * num_atoms, z_dim)
        zIG_std = self.zIG_std_enc(x)
        
        return Normal(zIG_mu, zIG_std), Normal(zIA_mu, zIA_std), temp_rnn_state

class Decoder(nn.Module):
    """
    Critic network class for HAPPO. Outputs value function predictions given centralized input (HAPPO) or local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, action_space, train_share_observation_space, obs_space, num_agents, device=torch.device("cpu")):
        super(Decoder, self).__init__()

        self.hidden_size = args.intention_hidden_size
        self.rnn_hidden_size = args.hidden_size
        self.intention_size = args.intention_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.args = args
        self._use_feature_normalization = args.use_feature_normalization
        self._gain = nn.init.calculate_gain('relu')
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_space = action_space
        self.num_agents = num_agents

        # if action_space.__class__.__name__ == 'Discrete':
        #     input_dim = action_space.n
        # else:
        self.action_dim = get_shape_from_act_space(action_space)
        self.input_dim = get_shape_from_obs_space(obs_space)[0]
        
        self.output_dim = get_shape_from_obs_space(train_share_observation_space)[0]
        
        if self._use_feature_normalization:             
            self.input_feature_norm = nn.LayerNorm(self.input_dim)
            self.output_feature_norm = nn.LayerNorm(self.output_dim)

        self.x_enc = MLPBase(args, get_shape_from_obs_space(obs_space))
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:             
            self.rnn = RNNLayer(self.rnn_hidden_size, self.rnn_hidden_size, self._recurrent_N, self._use_orthogonal)
        self.action_enc = nn.Linear(self.action_dim, self.hidden_size)
        
        # self.out_fc = nn.Sequential(nn.Linear(self.hidden_size + self.intention_size, self.hidden_size),
        #                                         nn.ReLU(),
        #                                         nn.Linear(self.hidden_size, self.input_dim))
        # # self.rnn = nn.GRU(input_dim, self.hidden_size, num_layers=self._recurrent_N)
        if not self.args.causal_inference_or_attn:
            self.intention_fc = nn.Linear(self.num_agents*(self.hidden_size + self.intention_size), self.hidden_size)
        else:
            self.intention_fc = nn.Linear(self.intention_size * self.num_agents, self.hidden_size)
        
        self.out_fc = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size, self.output_dim))

        self.attn_head = self.args.attn_head
        self.num_sample = self.args.num_sample
        if not self.args.causal_inference_or_attn:
            self.GAT = GraphAttentionConv(
                self.hidden_size, self.intention_size, self.attn_head, K=self.attn_head)

        self.to(device)

    def single_step_forward(self, graph, correlated_agents, rnn_states, masks, obs, actions):
        graph.ndata['xt_enc'] = self.action_enc(actions)
        obs_enc = self.x_enc(self.input_feature_norm(obs))
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:                          
            obs_enc, rnn_states = self.rnn(obs_enc, rnn_states, masks)
            obs_enc = obs_enc.reshape(-1, self.num_sample, obs_enc.shape[-1])
            rnn_states = rnn_states.reshape(-1, self.num_sample, self._recurrent_N, rnn_states.shape[-1])
        if not self.args.causal_inference_or_attn:
            graph = self.GAT.conv(graph)
            h = torch.cat([graph.ndata['deltax'], graph.ndata['zIA']], dim=-1)
            h = h.reshape(-1, self.num_agents, self.hidden_size + self.intention_size).permute(2, 0, 1) * correlated_agents.reshape(-1, correlated_agents.shape[-1])       
            # h = h.permute(1, 2, 0)         
            # h = h.view(h.shape[0], -1) # 32, 60
            # h = h.view(-1, self.num_sample, h.shape[-1])
            # h = torch.cat([h, obs_enc], dim=-1)
        else:
            h = graph.ndata['zIA'].reshape(-1, self.num_agents, self.intention_size).permute(2, 0, 1) * correlated_agents.reshape(-1, correlated_agents.shape[-1])                             
        h = h.permute(1, 2, 0)                
        # h = torch.cat([h, self.action_enc(actions).reshape(-1, self.num_agents, self.hidden_size)], dim=-1)      
        h = h.view(-1, self.num_sample, self.num_agents*(self.hidden_size + self.intention_size)) # 32, 60             h = torch.cat([h, graph.ndata[xt_enc']], dim=-1)
        # h = h.view(-1, self.num_sample, h.shape[-1])
        # h = torch.cat([h, obs_enc], dim=-1)
        h = self.intention_fc(h) + obs_enc
        deltax = self.out_fc(h)
        obs_enc = self.output_feature_norm(deltax)
        # graph.ndata['xt'] = graph.ndata['xt'] + deltax
        return graph, obs_enc, rnn_states

    def forward(self, graph, actions, inputs, correlated_agents=None, rnn_states=None, masks=None, pred_steps=1, forecast=False):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        if not self.args.causal_inference_or_attn:
            graph = self.GAT.attn(graph)
        inputs = check(inputs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        correlated_agents = check(correlated_agents).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        inputs = inputs.transpose(0, 1).contiguous() # [num_timesteps, batchsize * n_agents, num_dims]
        actions = actions.transpose(0, 1).contiguous() 
        masks = masks.transpose(0, 1).contiguous() # [num_timesteps, batchsize * n_agents, num_dims]
        correlated_agents = correlated_agents.transpose(0, 1).contiguous() # [num_timesteps, batchsize * n_agents, num_dims]
        # rnn_states = rnn_states.transpose(0, 1).contiguous() # [num_timesteps, batchsize * n_agents, num_dims]
        num_sample = graph.ndata['zIA'].shape[1]

        inputs = torch.stack([inputs] * num_sample, dim=2)
        actions = torch.stack([actions] * num_sample, dim=2)
        rnn_states = torch.stack([rnn_states] * num_sample, dim=2)
        correlated_agents = torch.stack([correlated_agents] * num_sample, dim=2)
        masks = torch.stack([masks] * num_sample, dim=2)

        sizes = inputs.shape
        # if self.action_space.__class__.__name__ == 'Discrete':
        #     # inputs = inputs.repeat(1, 1, self.action_space.n)
        #     inputs = inputs.reshape(-1, sizes[-1]).to(dtype=torch.int64).squeeze(-1)
        #     inputs = F.one_hot(inputs, num_classes=self.action_space.n).reshape(sizes[0], sizes[1], sizes[2], -1).to(**self.tpdv)
        #     if forecast:
        #         preds_out = torch.zeros((pred_steps, inputs.shape[1], inputs.shape[2], self.action_space.n)).to(**self.tpdv)
        #     else:
        #         preds_out = torch.zeros((*sizes[0:-1], self.action_space.n)).to(**self.tpdv)
        # else:
        if forecast:
            preds_out = torch.zeros(pred_steps, inputs.shape[1], inputs.shape[2],self.output_dim).to(**self.tpdv)
        else:
            preds_out = torch.zeros(sizes[0], sizes[1],sizes[2], self.output_dim).to(**self.tpdv)

        last_pred = inputs[0::pred_steps, :, :]
        last_action = actions[0::pred_steps, :, :]
        last_masks = masks[0::pred_steps, :, :]
        last_correlated_agents = correlated_agents[0::pred_steps, :, :]
        for b_idx in range(0, last_pred.shape[0]):
            obs = last_pred[b_idx]
            masks = last_masks[b_idx]
            actions = last_action[b_idx]
            correlated_agents = last_correlated_agents[b_idx]
            for step in range(0, pred_steps):
                graph, obs, rnn_states = self.single_step_forward(graph, correlated_agents, rnn_states, masks, obs, actions)
                # if self.action_space.__class__.__name__ == 'Discrete':
                #     if not forecast:
                #         tmp_x = torch.zeros(((inputs[step + b_idx * pred_steps, :, :, 0] == 1).sum(), self.action_space.n)).to(**self.tpdv)
                #         tmp_x[:, 0] = 1
                #         graph.ndata['xt'][inputs[step + b_idx * pred_steps, :, :, 0] == 1] = tmp_x
                #     preds_out[step + b_idx * pred_steps, :, :, :] =  graph.ndata['xt']
                # else:
                preds_out[step + b_idx * pred_steps, :, :] = obs
            if forecast:
                return obs, rnn_states

        return preds_out.transpose(0, 1).contiguous() # [batch_size * num_vars, num_timesteps, num_inputs]
