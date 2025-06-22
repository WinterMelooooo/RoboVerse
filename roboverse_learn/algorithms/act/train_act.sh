# Examples:
# bash roboverse_learn/algorithms/act/train_act.sh roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka CloseBoxFrankaL0 100 0 2000 ee ee

# 'metadata_dir' means path to metadata directory. e.g. roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka
# 'task_name' gives a name to the policy, which can include the task robot and level ie CloseBoxFrankaL0
# 'expert_data_num' means number of training data. e.g.100
# 'gpu_id' means single gpu id, e.g.0

metadata_dir=${1}
task_name=${2}
expert_data_num=${3}
gpu_ids=${4}

num_epochs=${5}
obs_space=${6} # joint_pos or ee
act_space=${7} # joint_pos or ee
delta_ee=${8:-0} # 0 or 1 (only matters if act_space is ee, 0 means absolute 1 means delta control )
config_name=${9:-"robot_act_rgb"}
master_port=${10:-50023}
seed=${11:-42}

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi



echo -e "\033[33mgpu id (to use): ${gpu_ids}\033[0m"
echo -e "master port: ${master_port}"
echo -e "seed: ${seed}"
NPROC=$(echo "${gpu_ids}" | tr ',' '\n' | wc -l)
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_ids}
torchrun --nproc_per_node=${NPROC} --nnodes=1 --master_port=${master_port} \
-m roboverse_learn.algorithms.act.train --config-name=${config_name}.yaml \
task_name=${task_name}_${extra} \
dataset.dataset_dir="data_policy/${task_name}_${extra}_${expert_data_num}.zarr" \
training.num_epochs=${num_epochs} \
training.seed=${seed} \
dataset.num_episodes=${expert_data_num} \
