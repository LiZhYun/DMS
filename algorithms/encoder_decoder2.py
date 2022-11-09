import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer2
from algorithms.utils.act import ACTLayer
from utils.util import get_shape_from_act_space
import numpy as np
from torch.distributions import Normal

def build_input(past, future=None):
    if future is not None:
        actions = np.concatenate([past, future], axis=1)
    else:
        actions = past
    actions = actions.transpose(0, 2, 1, 3)  # [bz, n_agents, timesteps, feat]
    actions = actions.reshape((-1, *actions.shape[2:])) # [batch_size * n_agents, timesteps, input_size]

    return actions

class Encoder_Decoder(nn.Module):
    def __init__(self, args, act_space, num_agents, device=torch.device("cpu")):
        super(Encoder_Decoder, self).__init__()
        self.encoder = Encoder(args, act_space, num_agents, device)
        self.decoder = Decoder(args, act_space, device)

class Encoder(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, action_space, num_agents, device=torch.device("cpu")):
        super(Encoder, self).__init__()
        self.hidden_size = args.intention_hidden_size
        self.intention_size = args.intention_size
        self.args=args
        self.num_agents = num_agents
        self._gain = nn.init.calculate_gain('relu')
        # self._use_orthogonal = True
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks  # by default True, whether to mask useless data in policy loss
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy # by default False, use the whole trajectory to calculate hidden states.
        self._use_recurrent_policy = args.use_recurrent_policy # by default, use Recurrent Policy. If set, do not use.
        self._recurrent_N = args.recurrent_N
        self._use_feature_normalization = args.use_feature_normalization
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._layer_N = args.layer_N * 2
        # base = MLPBase
        rnn_input_dim = get_shape_from_act_space(action_space)
        # self.base = base(args, [rnn_input_dim])
        # if self._use_naive_recurrent_policy or self._use_recurrent_policy:
        self.rnn = RNNLayer2(rnn_input_dim, self.hidden_size, self._recurrent_N, self._use_orthogonal, self.num_agents)
        # self.rnn = nn.GRU(rnn_input_dim, self.hidden_size, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self._gain = args.gain
        init_method = nn.init.xavier_normal_
        # init_method = [nn.init.xavier_normal_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0.1))
        
        if self._use_feature_normalization:
            self.input_feature_norm = nn.LayerNorm(rnn_input_dim)
            self.feature_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.zI_mu_enc1 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(self.hidden_size, self.hidden_size)), nn.ReLU()) for i in range(self._layer_N)])
        self.zI_mu_enc2 = nn.Sequential(init_(
            nn.Linear(self.hidden_size, self.intention_size)), nn.ReLU())
        self.zI_mu_enc3 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(self.intention_size, self.intention_size)), nn.ReLU()) for i in range(self._layer_N)])

        self.zI_std_enc1 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(self.hidden_size, self.hidden_size)), nn.Softplus()) for i in range(self._layer_N)])
        self.zI_std_enc2 = nn.Sequential(init_(
            nn.Linear(self.hidden_size, self.intention_size)), nn.Softplus())
        self.zI_std_enc3 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(self.intention_size, self.intention_size)), nn.Softplus()) for i in range(self._layer_N)])

        self.to(device)

    def forward(self, actions, hidden=None):
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

        if self._use_feature_normalization:
            x = self.input_feature_norm(actions)
        # x = self.base(actions)
        if hidden is None:
            _, temp_rnn_state = self.rnn(x, hidden)  # (1, batchsize * n_agents, hidden_size)
        else:
            hidden = check(hidden).to(**self.tpdv)
            _, temp_rnn_state = self.rnn(actions, hidden)
        x = temp_rnn_state.squeeze(0)
        x = self.feature_norm(x)
        x = self.dropout(x)
        
        zI_mu = x.clone()
        for i in range(self._layer_N):
            zI_mu = self.zI_mu_enc1[i](zI_mu)
        zI_mu = zI_mu + x

        zI_mu = self.zI_mu_enc2(zI_mu)

        zI_mu_i = zI_mu.clone()
        for i in range(self._layer_N):
            zI_mu = self.zI_mu_enc3[i](zI_mu)
        zI_mu = zI_mu + zI_mu_i
        
        zI_std = x.clone()
        for i in range(self._layer_N):
            zI_std = self.zI_std_enc1[i](zI_std)
        zI_std = zI_std + x

        zI_std = self.zI_std_enc2(zI_std)

        zI_std_i = zI_std.clone()
        for i in range(self._layer_N):
            zI_std = self.zI_std_enc3[i](zI_std)
        zI_std = zI_std + zI_std_i
        
        return Normal(zI_mu, zI_std), temp_rnn_state


class Decoder(nn.Module):
    """
    Critic network class for HAPPO. Outputs value function predictions given centralized input (HAPPO) or local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, action_space, device=torch.device("cpu")):
        super(Decoder, self).__init__()
        self.hidden_size = args.intention_hidden_size
        self.intention_size = args.intention_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.args = args
        self._use_feature_normalization = args.use_feature_normalization
        self._gain = nn.init.calculate_gain('relu')
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = nn.init.xavier_normal_
        # init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        self.action_space = action_space
        self._layer_N = args.layer_N * 2
        if action_space.__class__.__name__ == 'Discrete':
            input_dim = action_space.n
        else:
            input_dim = get_shape_from_act_space(action_space)
        
        if self._use_feature_normalization:             
            self.input_feature_norm = nn.LayerNorm(input_dim)
            self.feature_norm = nn.LayerNorm(self.hidden_size)
        
        self.dropout = nn.Dropout(0.5)

        self.rnn = nn.GRU(input_dim, self.hidden_size, num_layers=self._recurrent_N)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0.1))
        # if action_space.__class__.__name__ == 'Discrete':
        self.x_enc = nn.ModuleList([nn.Sequential(init_(                 
            nn.Linear(self.hidden_size, self.hidden_size)), nn.ReLU()) for i in range(self._layer_N)])
        self.out_fc1 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(self.hidden_size + self.intention_size, self.hidden_size + self.intention_size)), nn.ReLU()) for i in range(self._layer_N)])
        self.out_fc2 = nn.Sequential(init_(
            nn.Linear(self.hidden_size + self.intention_size, self.hidden_size)), nn.ReLU())
        self.out_fc3 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(self.hidden_size, self.hidden_size)), nn.ReLU()) for i in range(self._layer_N)])
        self.out_fc4 = nn.Sequential(init_(
            nn.Linear(self.hidden_size, input_dim)), nn.ReLU())
        # else:

        #     self.x_enc = nn.ModuleList([nn.Sequential(init_(
        #         nn.Linear(self.hidden_size, self.hidden_size)), nn.ReLU(), nn.LayerNorm(self.hidden_size)) for i in range(self._layer_N)])
        #     self.out_fc1 = nn.ModuleList([nn.Sequential(init_(
        #         nn.Linear(self.hidden_size + self.intention_size, self.hidden_size + self.intention_size)), nn.ReLU(), nn.LayerNorm(self.hidden_size + self.intention_size)) for i in range(self._layer_N)])
        #     self.out_fc2 = nn.Sequential(init_(
        #         nn.Linear(self.hidden_size + self.intention_size, self.hidden_size)), nn.ReLU(), nn.LayerNorm(self.hidden_size))
        #     self.out_fc3 = nn.ModuleList([nn.Sequential(init_(
        #         nn.Linear(self.hidden_size, self.hidden_size)), nn.ReLU(), nn.LayerNorm(self.hidden_size)) for i in range(self._layer_N)])
        #     self.out_fc4 = nn.Sequential(init_(
        #         nn.Linear(self.hidden_size, input_dim)), nn.ReLU(), nn.LayerNorm(input_dim))

        self.to(device)

    def forward(self, intention, inputs, pred_steps=20):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        inputs = check(inputs).to(**self.tpdv)
        intention = check(intention).to(**self.tpdv)
        inputs = inputs.transpose(0, 1).contiguous() # [num_timesteps, batchsize * n_agents, num_dims]

        inputs = torch.stack([inputs] * 10, dim=2)

        sizes = inputs.shape
        if self.action_space.__class__.__name__ == 'Discrete':
            # inputs = inputs.repeat(1, 1, self.action_space.n)
            inputs = inputs.reshape(-1, sizes[-1]).to(dtype=torch.int64).squeeze(-1)
            inputs = F.one_hot(inputs, num_classes=self.action_space.n).reshape(sizes[0], sizes[1], sizes[2], -1).to(**self.tpdv)
            
            preds_out = torch.zeros((*sizes[0:-1], self.action_space.n)).to(**self.tpdv)
        else:
            preds_out = torch.zeros(sizes).to(**self.tpdv)

        hidden_states = None
        last_pred = inputs[0::pred_steps, :, :]
        for b_idx in range(0, last_pred.shape[0]):
            x = last_pred[b_idx]
            for step in range(0, pred_steps):
                xt_enc = x.reshape((-1, *x.shape[2:])).unsqueeze(0)
                if self._use_feature_normalization:             
                    xt_enc = self.input_feature_norm(xt_enc)
                _, hidden_states = self.rnn(xt_enc, hidden_states)
                xt_enc = hidden_states.squeeze(0).reshape(*x.shape[0:2], -1)
                xt_enc = self.feature_norm(xt_enc)

                input_0 = xt_enc.clone()
                for i in range(self._layer_N):
                    xt_enc = self.x_enc[i](xt_enc)
                xt_enc = xt_enc + input_0
                # xt_enc = self.dropout(xt_enc)
                h = torch.cat([xt_enc, intention], dim=-1)
                h = self.dropout(h)
                
                if self.action_space.__class__.__name__ == 'Discrete':
                    input1 = h.clone()
                    for i in range(self._layer_N):
                        h = self.out_fc1[i](h)
                    h = h + input1
                    h = self.out_fc2(h)
                    input2 = h.clone()
                    for i in range(self._layer_N):
                        h = self.out_fc3[i](h)
                    h = h + input2
                    deltax = self.out_fc4(h)
                    x = x + deltax
                    tmp_x = torch.zeros(((inputs[step + b_idx * pred_steps, :, :, 0] == 1).sum(), self.action_space.n)).to(**self.tpdv)
                    tmp_x[:, 0] = 1
                    x[inputs[step + b_idx * pred_steps, :, :, 0] == 1] = tmp_x
                    # x[inputs[step + b_idx * pred_steps, :, 0] == 1] = torch.zeros(((inputs[step + b_idx * pred_steps, :, :] == 0.).sum())).to(**self.tpdv)
                    preds_out[step + b_idx * pred_steps, :, :, :] = x
                else:
                    input1 = h.clone()
                    for i in range(self._layer_N):
                        h = self.out_fc1[i](h)
                    h = h + input1
                    h = self.out_fc2(h)
                    input2 = h.clone()
                    for i in range(self._layer_N):
                        h = self.out_fc3[i](h)
                    h = h + input2
                    deltax = self.out_fc4(h)
                    # deltax = self.out_fc(h)  # [batchsize * n_agents, num_dims]
                    x = x + deltax
                    preds_out[step + b_idx * pred_steps, :, :] = x

        return preds_out.transpose(0, 1).contiguous() # [batch_size * num_vars, num_timesteps, num_inputs]
