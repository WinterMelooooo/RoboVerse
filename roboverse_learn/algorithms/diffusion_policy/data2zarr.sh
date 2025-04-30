# Examples:
# bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka CloseBoxFrankaL0 100 0 200 joint_pos joint_pos

# 'metadata_dir' means path to metadata directory. e.g. roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka
# 'task_name' gives a name to the policy, which can include the task robot and level ie CloseBoxFrankaL0
# 'expert_data_num' means number of training data. e.g.100
# 'gpu_ids' means single gpu id, e.g.0

metadata_dir=${1}
task_name=${2}
expert_data_num=${3}
gpu_ids=${4}
num_epochs=${5}
obs_space=${6} # joint_pos or ee
act_space=${7} # joint_pos or ee
delta_ee=${8:-0} # 0 or 1 (only matters if act_space is ee, 0 means absolute 1 means delta control )
store_rgbd=${9:-0} # 0 or 1
store_pnt_cloud=${10:-0} # 0 or 1
tag="${11:-}" # the number of name of checkpoint, e.g. 200 for 200.ckpt
output_dir=${12:-} # the output directory, e.g. /home/ghr/yktang/RoboVerse/info/outputs/DP/2025.04.20/16.43.28_CloseBoxFrankaL0_obs:joint_pos_act:joint_pos

horizon=8
n_obs_steps=3
n_action_steps=4
seed=42

# adding the obs and action space as additional info
extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

python roboverse_learn/algorithms/data2zarr_dp.py \
--task_name ${task_name}_${extra} \
--expert_data_num ${expert_data_num} \
--metadata_dir ${metadata_dir} \
--action_space ${act_space} \
--observation_space ${obs_space} \
--delta_ee ${delta_ee} \
--store_rgbd ${store_rgbd} \
--store_pnt_cloud ${store_pnt_cloud}
