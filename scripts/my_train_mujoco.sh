#!/bin/sh

# # seed 6688 8630
# env="mujoco"
# scenario="HalfCheetah-v2"
# agent_conf="3x2"
# agent_obsk=0
# algo="happo"
# exp="dms-3x2"
# running_max=5
# echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${running_max}"
# for number in `seq ${running_max}`;
# do
#     echo "the ${number}-th running:"
#     python train/train_mujoco.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --running_id ${number} --use_intention --long_short_clip --intention_clip
# done

env="mujoco"
scenario="HalfCheetah-v2"
agent_conf="6x1"
agent_obsk=1
algo="happo"
exp="dms-6x1"
running_max=1
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for number in `seq ${running_max}` #12321
do
    echo "the ${number}-th running:"
    python -W ignore train/train_mujoco.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --running_id ${number} --use_intention --long_short_clip --cuda_deterministic #--use_linear_lr_decay #--use_centralized_V #--use_linear_lr_decay #--use_gae #--seed_specify --intention_clip --use_value_active_masks --use_eval --add_center_xy --use_state_agent --share_policy
done