import time
from turtle import st
import numpy as np
from functools import reduce
import torch
from runners.separated.base_runner import Runner
from algorithms.utils.util import check

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for i in range(self.num_agents):                     
                    self.trainer[i].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_intention = self.collect(step)
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions, all_last_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_intention, all_last_actions
                
                # insert data into buffer
                self.insert(data, step, episode)

            # compute return and update network
            self.compute()
            train_infos = self.train(episode)
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Seed {} Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.seed,
                                self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []                    

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won
                # modified

                for agent_id in range(self.num_agents):
                    train_infos[agent_id]['dead_ratio'] = 1 - self.buffer[agent_id].active_masks.sum() /(self.num_agents* reduce(lambda x, y: x*y, list(self.buffer[agent_id].active_masks.shape)))
                
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions, all_last_actions = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:,agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:,agent_id].copy()
            self.buffer[agent_id].available_actions[0] = available_actions[:,agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector=[]
        action_collector=[]
        action_log_prob_collector=[]
        rnn_state_collector=[]
        rnn_states_intention_collector=[]
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_states_intention \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                self.buffer[agent_id].obs[step],
                                                self.buffer[agent_id].rnn_states[step],
                                                self.buffer[agent_id].masks[step],
                                                available_actions=self.buffer[agent_id].available_actions[step],                                                           
                                                use_intention=self.use_intention,                                                              
                                                correlated_agents=self.buffer[agent_id].correlated_agents[step],
                                                rnn_states_intention=self.buffer[agent_id].rnn_states_intention[step],                                                             
                                                last_actions=self.buffer[agent_id].last_actions[step])
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_states_intention_collector.append(_t2n(rnn_states_intention))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_intention = np.array(rnn_states_intention_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_intention

    def insert(self, data, step, episode):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_intention, all_last_actions = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_intention[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_intention.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        correlated_agents = np.ones((self.n_rollout_threads, self.num_agents, self.num_agents), dtype=np.float32) 

        train_share_obs = share_obs
        if not self.use_centralized_V:
            share_obs = obs

        # for agent_id in range(self.num_agents):
        #     actions_tmp = check(np.array(all_last_actions[:, agent_id, :])).to(**self.tpdv).unsqueeze(2) # 1, 4, 1, 2
        #     actions_tmp = actions_tmp.view(-1, 1, self.envs.action_dim)
        #     intention_rnn_state = check(self.buffer[agent_id].rnn_states_intention[step]).to(**self.tpdv).reshape(self.n_rollout_threads*self.num_agents, self.recurrent_N, self.intention_hidden_size).transpose(0, 1) # batchsize * n_agents, 1, 64 -> 1, batchsize * n_agents, 64
        #     current_intention, temp_intention_rnn_state = self.policy[agent_id].encoder_decoder.encoder(actions_tmp, intention_rnn_state)
        #     rnn_states_intention[:, agent_id] = _t2n(temp_intention_rnn_state.transpose(0, 1).reshape(self.n_rollout_threads, self.num_agents, self.recurrent_N, self.intention_hidden_size)) # (1, batchsize * n_agents, hidden_size) - > (batchsize * n_agents, 1, hidden_size)
        #     intentions[:, agent_id] = _t2n(current_intention.mean).reshape(self.n_rollout_threads, self.num_agents, self.intention_size)
        #     # if self.use_intention and episode >  100 and (episode * self.episode_length * self.n_rollout_threads + step) % 10 == 0:
        #     #     actor_features_obs = self.policy[agent_id].actor.base(check(obs[:, agent_id]).to(**self.tpdv))
        #     #     # actor_features_obs, _ = self.policy[agent_id].actor.rnn(actor_features_obs, check(rnn_states[:, agent_id]).to(**self.tpdv), check(masks[:, agent_id]).to(**self.tpdv))
        #     #     correlated_agents[:, agent_id] = self.policy[agent_id].Correlated_Agents(intentions[:, agent_id], None, actor_features_obs, check(rnn_states[:,agent_id]).to(**self.tpdv), check(masks[:,agent_id]).to(**self.tpdv))
        #     # else:                 
        #     #     correlated_agents[:, agent_id] = self.buffer[agent_id].correlated_agents[step]

        # rnn_states_intention[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.num_agents, self.recurrent_N, self.intention_hidden_size), dtype=np.float32)


        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:,agent_id], obs[:,agent_id], rnn_states[:,agent_id],
                    rnn_states[:,agent_id],actions[:,agent_id], action_log_probs[:,agent_id],
                    values[:,agent_id], rewards[:,agent_id], masks[:,agent_id], 
                    rnn_states_intention[:, agent_id], rnn_states_intention[:, agent_id], correlated_agents[:, agent_id], all_last_actions[:, agent_id, :],
                    bad_masks[:,agent_id], active_masks[:,agent_id], available_actions[:,agent_id], train_share_obs=train_share_obs[:, agent_id])

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
            agent_k = "agent%i/" % agent_id + 'vae_lr'             
            self.writter.add_scalars(agent_k, {agent_k: self.trainer[agent_id].policy.encoder_decoder_optimizer.param_groups[0]['lr']}, total_num_steps) 
        # if ((total_num_steps - 3200) / 16000 ) % 100 == 0:
        #     mat = np.stack([self.buffer[i].correlated_agents[-1][0] for i in range(self.num_agents)])
        #     self.writter.add_embedding(mat, self.meta_data, global_step=total_num_steps, tag="correlated_agents")
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])

        eval_obs, eval_share_obs, eval_available_actions, all_last_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        
        eval_intention_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.num_agents, self.recurrent_N, self.intention_hidden_size), dtype=np.float32)          
        # intentions_eval = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.num_agents, self.intention_size), dtype=np.float32)                  
        # intentions_eval = np.random.normal(0, 1, size=(self.n_eval_rollout_threads, self.num_agents, self.num_agents, self.intention_size)) 
        correlated_agents_eval = np.ones((self.n_eval_rollout_threads, self.num_agents, self.num_agents), dtype=np.float32) 
        eval_dones_env_all = [False] * self.n_eval_rollout_threads 

        steps = 0
        while True:
            eval_actions_collector=[]
            eval_rnn_states_collector=[]
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state, rnn_states_intention = \
                    self.trainer[agent_id].policy.act(eval_obs[:,agent_id],
                                            eval_rnn_states[:,agent_id],
                                            eval_masks[:,agent_id],
                                            eval_available_actions[:,agent_id],
                                            deterministic=True,
                                            rnn_states_intention=eval_intention_rnn_states[:, agent_id],                                              
                                            last_actions=all_last_actions,                                                 
                                            correlated_agents=correlated_agents_eval[:, agent_id],                                                        
                                            )
                eval_rnn_states[:,agent_id]=_t2n(temp_rnn_state)
                eval_intention_rnn_states[:, agent_id] = _t2n(rnn_states_intention)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1,0,2)

            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions, all_last_actions = self.eval_envs.step(eval_actions)

            # if self.use_intention:
            #     for agent_id in range(self.num_agents):
            #         actions = check(np.array(all_last_actions[:, agent_id, :])).to(**self.tpdv).unsqueeze(2) # 1, 4, 1, 2
            #         actions = actions.view(-1, 1, self.envs.action_dim)
            #         intention_rnn_state = check(eval_intention_rnn_states[:, agent_id]).to(**self.tpdv).reshape(self.n_eval_rollout_threads*self.num_agents, self.recurrent_N, self.intention_hidden_size).transpose(0, 1) # batchsize * n_agents, 1, 64 -> 1, batchsize * n_agents, 64
            #         current_intention, temp_intention_rnn_state = self.policy[agent_id].encoder_decoder.encoder(actions, intention_rnn_state)
            #         eval_intention_rnn_states[:, agent_id] = _t2n(temp_intention_rnn_state.transpose(0, 1).reshape(self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.intention_hidden_size)) # (1, batchsize * n_agents, hidden_size) - > (batchsize * n_agents, 1, hidden_size)
            #         intentions_eval[:, agent_id] = _t2n(current_intention.mean).reshape(self.n_eval_rollout_threads, self.num_agents, self.intention_size)
            #         # if steps % 19 == 0:
            #         #     actor_features_obs = self.policy[agent_id].actor.base(check(eval_obs[:, agent_id]).to(**self.tpdv)) 
            #         #     correlated_agents_eval[:, agent_id] = self.policy[agent_id].Correlated_Agents(intentions_eval[:, agent_id], None, actor_features_obs, check(eval_rnn_states[:,agent_id]).to(**self.tpdv), check(eval_masks[:,agent_id]).to(**self.tpdv))
            #         # if ((total_num_steps - 3200) / 16000 ) % 100 == 0:
                        
            steps += 1

            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_intention_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.num_agents, self.recurrent_N, self.intention_hidden_size), dtype=np.float32)  

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i] and eval_dones_env_all[eval_i] == False:
                    # print( steps)
                    # steps = 0
                    eval_dones_env_all[eval_i] = True
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.concatenate(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                # mat = np.stack([correlated_agents_eval[0][i] for i in range(self.num_agents)])
                # self.writter.add_embedding(mat, self.meta_data, global_step=total_num_steps, tag="correlated_agents")
                self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
