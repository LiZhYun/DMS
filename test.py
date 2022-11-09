# meta_data = ['Agent_{}'.format(i) for i in range(10)]
# print(meta_data)
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

eval_average_episode_rewards_all = []

with open('scripts/results/mujoco/HalfCheetah-v2/happo/dms/long_short_clip/8888/run6/logs/summary.json', 'r+') as f:
    exp_dict = json.load(f)
    eval_average_episode_rewards = exp_dict["/home/lizhiyuan/code/Coordination-in-multi-agent-2.0/scripts/results/mujoco/HalfCheetah-v2/happo/dms/long_short_clip/8888/run6/logs/eval_average_episode_rewards/eval_average_episode_rewards"]
    eval_average_episode_rewards_all.append(eval_average_episode_rewards)
with open('scripts/results/mujoco/HalfCheetah-v2/happo/dms/long_short_clip/8888/run9/logs/summary.json', 'r+') as f:
    exp_dict = json.load(f)
    eval_average_episode_rewards = exp_dict["/home/lizhiyuan/code/Coordination-in-multi-agent-2.0/scripts/results/mujoco/HalfCheetah-v2/happo/dms/long_short_clip/8888/run9/logs/eval_average_episode_rewards/eval_average_episode_rewards"]
    eval_average_episode_rewards_all.append(eval_average_episode_rewards)
with open('scripts/results/mujoco/HalfCheetah-v2/happo/dms/long_short_clip/8888/run10/logs/summary.json', 'r+') as f:
    exp_dict = json.load(f)
    eval_average_episode_rewards = exp_dict["/home/lizhiyuan/code/Coordination-in-multi-agent-2.0/scripts/results/mujoco/HalfCheetah-v2/happo/dms/long_short_clip/8888/run10/logs/eval_average_episode_rewards/eval_average_episode_rewards"]
    eval_average_episode_rewards_all.append(eval_average_episode_rewards)

eval_average_episode_rewards_all = np.stack(eval_average_episode_rewards_all)[:, :, 1:]
time = eval_average_episode_rewards_all[0, :, 0]
data = eval_average_episode_rewards_all[:, :, 1]

def smooth(data, sm=2):
    if sm > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(sm)
        for i in range(data.shape[0]):
            x = np.asarray(data[i])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            data[i] = smoothed_x

    return data
# df = pd.DataFrame({
#     'Environment steps': eval_average_episode_rewards_all[:, :, 0].flatten(),
#     'Average Episode Reward': eval_average_episode_rewards_all[:, :, 1].flatten()
# })
# pd.DataFrame({'date': ['1/1/2021',
#                             '1/2/2021',
#                             '1/3/2021',
#                             '1/4/2021',
#                             '1/1/2021',
#                             '1/2/2021',
#                             '1/3/2021',
#                             '1/4/2021'],
#                    'sales': [4, 7, 8, 13, 17, 15, 21, 28],
#                    'company': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']})
data = smooth(data)
# fig = plt.figure()
linestyle = ['-', '--', ':', '-.'] 
color = ['r', 'g', 'b', 'k'] 
label = ['algo1', 'algo2', 'algo3', 'algo4']
for i in range(1):         
    sns.tsplot(time=time, data=data, color=color[i], linestyle=linestyle[i], condition=label[i])
# sns.lmplot(x='Environment steps',y='Average Episode Reward',data=df,fit_reg=False)
# sns.lineplot(x='date', y='value', data=df)
plt.ylabel("Average Episode Reward", fontsize=25) 
plt.xlabel("Environment steps", fontsize=25) 
plt.title("2x3-Agent HalfCheetah", fontsize=30) 
plt.show()
