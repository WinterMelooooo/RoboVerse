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
config_name=${11:-"robot_dp"}
master_port=${12:-50023}
max_visible_ratio=${13:-100} # 0.5 for 50% visible, 1 for 100% visible
multigpu_lr_policy=${14:-"sqrt"}
seed=${15:-42}
logger=${16:-"wandb"} # tensorboard or wandb
consistent_warmup=${17:-1} # 1 for true, 0 for false
backend=${18:-1} # 1 for true, 0 for false
cotrain=${19:-0} # 1 for true, 0 for false
real_world_ratio=${20:-100} # 100 for 100% real world data,
real_world_task_name=${21:-} # path to real world zarr data, e.g. data_policy/LiberoPickButter_obs:joint_pos_act:joint_pos_100.zarr
real_world_data_num=${22:-50} # number of real world data, e.g. 100
horizon=${23:-8} # 8 for 8 steps, 16 for 16 steps
n_obs_steps=${24:-3} # 3 for 3 steps, 2 for 2 steps
n_action_steps=${25:-4} # 4 for 4 steps, 8 for 8 steps
tag="${26:-}" # the number of name of checkpoint, e.g. 200 for 200.ckpt
output_dir=${27:-} # the output directory, e.g. /home/ghr/yktang/RoboVerse/info/outputs/DP/2025.04.20/16.43.28_CloseBoxFrankaL0_obs:joint_pos_act:joint_pos


# adding the obs and action space as additional info
extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

#python roboverse_learn/algorithms/data2zarr_dp.py \
#--task_name ${task_name}_${extra} \
#--expert_data_num ${expert_data_num} \
#--metadata_dir ${metadata_dir} \
#--action_space ${act_space} \
#--observation_space ${obs_space} \
#--delta_ee ${delta_ee} \
#--store_rgbd ${store_rgbd} \
#--store_pnt_cloud ${store_pnt_cloud}

echo -e "\033[33mgpu id (to use): ${gpu_ids}\033[0m"
echo -e "master port: ${master_port}"
echo -e "seed: ${seed}"
NPROC=$(echo "${gpu_ids}" | tr ',' '\n' | wc -l)
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_ids}
python -m torch.distributed.run --nproc_per_node=${NPROC} --nnodes=1 --master_port=${master_port} \
roboverse_learn/algorithms/diffusion_policy/train.py --config-name=${config_name}.yaml \
task.name="${task_name}_${extra}" \
task.dataset.zarr_path="data_policy/${task_name}_${extra}_${expert_data_num}.zarr" \
training.seed=${seed} \
horizon=${horizon} \
n_obs_steps=${n_obs_steps} \
n_action_steps=${n_action_steps} \
training.num_epochs=${num_epochs} \
policy_runner.obs.obs_type=${obs_space} \
policy_runner.action.action_type=${act_space} \
policy_runner.action.delta=${delta_ee} \
training.output_dir=${output_dir} \
training.tag=${tag} \
++backend=${backend} \
++task.dataset.max_visible_ratio=${max_visible_ratio} \
++optimizer.multigpu_lr_policy=${multigpu_lr_policy} \
++logging.logger_name=${logger} \
++training.consistent_warmup=${consistent_warmup} \
++task.dataset.cotraining=${cotrain} \
++task.dataset.real_world_ratio=${real_world_ratio} \
++task.dataset.real_world_zarr_path="data_policy/${real_world_task_name}_${extra}_${real_world_data_num}.zarr" \
