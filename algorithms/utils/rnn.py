import torch
import torch.nn as nn

"""RNN modules."""


class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        x = x.reshape(-1, x.shape[-1])
        hxs = hxs.reshape(-1, self._recurrent_N, hxs.shape[-1])
        masks = masks.reshape(-1, masks.shape[-1])
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(x.unsqueeze(0),
                              (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs


class RNNLayer2(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal, num_agents):
        super(RNNLayer2, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal
        self.num_agents = num_agents
        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        # for name, param in self.rnn.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0)
        #     elif 'weight' in name:
        #         if self._use_orthogonal:
        #             nn.init.orthogonal_(param)
        #         else:
        #             nn.init.xavier_uniform_(param)
        # self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs):
        if x.size(1) == hxs.size(1):
            if hxs != None:
                x, hxs = self.rnn(x, hxs)
            else:
                x, hxs = self.rnn(x)
        else: # 10， 800， 3   
            #1， 800， 64
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            
            # x = x.view(T, -1, self.num_agents, x.size(-1)) # 10, 400, 2, 3
            # hxs = hxs.view(self._recurrent_N, -1, self.num_agents, hxs.size(-1)) # 1, 400, 2, 64
            N = hxs.size(1)
            T = int(x.size(1) / N)
            x = x.view(T, -1, x.size(-1)) # 10, 400, 2, 3

            # add t=0 and t=T to the list
            has_zeros = list(range(T)) + [T]

            outputs = []
            for i in range(len(has_zeros)-1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(hxs)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            hxs = torch.cat(outputs, dim=0)

            # flatten
            hxs = hxs.reshape(T * N, -1)
            # hxs = hxs.transpose(0, 1)

        # x = self.norm(x)
        return None, hxs
