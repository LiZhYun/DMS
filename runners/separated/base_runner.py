    
from re import A
import time
import os
import numpy as np
from itertools import chain
from scipy import rand
import torch
from tensorboardX import SummaryWriter
from utils.separated_buffer import SeparatedReplayBuffer
from utils.util import update_linear_schedule, get_shape_from_obs_space
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        self.meta_data = ['Agent_{}'.format(i) for i in range(self.num_agents)]

        # parameters

        self.long_short_clip = self.all_args.long_short_clip
        self.intention_clip = self.all_args.intention_clip
        # self.intention_size = self.all_args.intention_size
        # self.intention_size = self.all_args.intention_hidden_size + self.all_args.intention_size
        self.intention_size = get_shape_from_obs_space(self.envs.share_observation_space[0])
        self.use_intention = self.all_args.use_intention
        self.tpdv = dict(dtype=torch.float32, device=self.device)

        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.intention_hidden_size = self.all_args.intention_hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.use_single_network = self.all_args.use_single_network
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)


        if self.all_args.algorithm_name == "happo":
            from algorithms.happo_trainer import HAPPO as TrainAlgo
            from algorithms.happo_policy import HAPPO_Policy as Policy
        elif self.all_args.algorithm_name == "hatrpo":
            from algorithms.hatrpo_trainer import HATRPO as TrainAlgo
            from algorithms.hatrpo_policy import HATRPO_Policy as Policy
        else:
            raise NotImplementedError

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            train_share_observation_space = self.envs.share_observation_space[agent_id]
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        self.num_agents,
                        train_share_observation_space,
                        device = self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm 
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id],
                                       self.envs.n_agents, 
                                       self.envs.action_dim,
                                       train_share_observation_space)
            self.buffer.append(bu)

            tr = TrainAlgo(self.all_args, self.policy[agent_id], self.num_agents, self.intention_size, device = self.device)
            self.trainer.append(tr)
            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1],
                                                                rnn_states_intention=self.buffer[agent_id].rnn_states_intention[-1],
                                                                last_actions=self.buffer[agent_id].last_actions[-1],
                                                                correlated_agents=self.buffer[agent_id].correlated_agents[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self, episode):
        train_infos = [{} for _ in range(self.num_agents)]
        # random update order

        action_dim=self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        # last_actions = np.zeros((self.episode_length, self.n_rollout_threads, self.num_agents, action_dim), dtype=np.float32) 
        randomseq = torch.randperm(self.num_agents)
        # factor = np.ones((self.episode_length, self.n_rollout_threads, action_dim), dtype=np.float32) 
        for num, agent_id in enumerate(randomseq):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            # self.buffer[agent_id].update_action(last_actions, randomseq[:num])
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])
            
            if self.all_args.algorithm_name == "hatrpo":
                old_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]),
                                                            encoder_decoder=self.trainer[agent_id].policy.encoder_decoder, 
                                                            use_intention=self.use_intention, 
                                                            correlated_agents=self.buffer[agent_id].correlated_agents[:-1].reshape(-1, *self.buffer[agent_id].correlated_agents.shape[2:]), 
                                                            rnn_states_intention=self.buffer[agent_id].rnn_states_intention[0:1].reshape(-1, *self.buffer[agent_id].rnn_states_intention.shape[2:]), 
                                                            last_actions=self.buffer[agent_id].last_actions.reshape(-1, *self.buffer[agent_id].last_actions.shape[2:])
                                                            )
            else:
                _, old_actions_logprob, _ =self.trainer[agent_id].policy.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]),
                                                            use_intention=self.use_intention,                                                              
                                                            correlated_agents=self.buffer[agent_id].correlated_agents[:-1].reshape(-1, *self.buffer[agent_id].correlated_agents.shape[2:]),                                                              
                                                            rnn_states_intention=self.buffer[agent_id].rnn_states_intention[0:1].reshape(-1, *self.buffer[agent_id].rnn_states_intention.shape[2:]),                                                              
                                                            last_actions=self.buffer[agent_id].last_actions.reshape(-1, *self.buffer[agent_id].last_actions.shape[2:]),
                                                            # encoder_decoder=self.trainer[agent_id].policy.encoder_decoder
                                                            )
            train_info = self.trainer[agent_id].train(self.buffer[agent_id], intention_clip=self.intention_clip, long_short_clip=self.long_short_clip, use_intention=self.use_intention, episode=episode)

            if self.all_args.algorithm_name == "hatrpo":
                new_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                _, new_actions_logprob, _ =self.trainer[agent_id].policy.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]),
                                                            use_intention=self.use_intention,                                                              
                                                            correlated_agents=self.buffer[agent_id].correlated_agents[:-1].reshape(-1, *self.buffer[agent_id].correlated_agents.shape[2:]),                                                              
                                                            rnn_states_intention=self.buffer[agent_id].rnn_states_intention[0:1].reshape(-1, *self.buffer[agent_id].rnn_states_intention.shape[2:]),                                                              
                                                            last_actions=self.buffer[agent_id].last_actions.reshape(-1, *self.buffer[agent_id].last_actions.shape[2:]),
                                                            # encoder_decoder=self.trainer[agent_id].policy.encoder_decoder
                                                            )
            # actions, _, _ = self.trainer[agent_id].policy.actor(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
            #                                                 self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
            #                                                 self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
            #                                                 available_actions,
            #                                                 intention=self.buffer[agent_id].intentions[:-1].reshape(-1, *self.buffer[agent_id].intentions.shape[2:]),                                                                
            #                                                 use_intention=self.use_intention,                                                              
            #                                                 correlated_agents=self.buffer[agent_id].correlated_agents[:-1].reshape(-1, *self.buffer[agent_id].correlated_agents.shape[2:]),                                                              
            #                                                 # rnn_states_intention=self.buffer[agent_id].rnn_states_intention[:-1].reshape(-1, *self.buffer[agent_id].rnn_states_intention.shape[2:]),                                                              
            #                                                 # last_actions=self.buffer[agent_id].last_actions.reshape(-1, *self.buffer[agent_id].last_actions.shape[2:]),
            #                                                 # encoder_decoder=self.trainer[agent_id].policy.encoder_decoder
            #                                                 )
            # last_actions[:, :, agent_id] = actions.view(self.episode_length, self.n_rollout_threads, -1).detach().cpu().numpy()
            factor = factor*_t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            # # factor = factor*_t2n(torch.exp(new_actions_logprob-old_actions_logprob).reshape(self.episode_length,self.n_rollout_threads,action_dim)) 
            train_infos[agent_id] = train_info
            self.buffer[agent_id].after_update()

        # for agent_id in randomseq:
        #     self.buffer[agent_id].update_action(last_actions)
        #     train_info = self.trainer[agent_id].train_vae(self.buffer[agent_id], intention_clip=self.intention_clip, long_short_clip=self.long_short_clip, use_intention=self.use_intention, episode=episode) 
        #     train_infos[agent_id].update(train_info)

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent" + str(agent_id) + ".pt")
            else:
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
                intention_model = self.trainer[agent_id].policy.encoder_decoder                 
                torch.save(intention_model.state_dict(), str(self.save_dir) + "/intention_agent" + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
                intention_model_state_dict = torch.load(str(self.model_dir) + '/intention_agent' + str(agent_id) + '.pt')                 
                self.policy[agent_id].encoder_decoder.load_state_dict(intention_model_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
