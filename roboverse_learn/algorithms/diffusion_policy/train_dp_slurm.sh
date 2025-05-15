#!/usr/bin/env bash
# train_dp_slurm.sh — 在 Slurm 分配的节点环境下启动多机多卡训练

# 参数传递与解析（同原 train_dp.sh）
metadata_dir=${1}
task_name=${2}
expert_data_num=${3}
gpu_ids=${4}
num_epochs=${5}
obs_space=${6}
act_space=${7}
delta_ee=${8:-0}
store_rgbd=${9:-0}
store_pnt_cloud=${10:-0}
config_name=${11:-"robot_dp"}
test_rescale=${12:-0}
max_visible_ratio=${13:-100}
multigpu_lr_policy=${14:-"sqrt"}
horizon=${15:-8}
n_obs_steps=${16:-3}
n_action_steps=${17:-4}
tag="${18:-}"
output_dir=${19:-}
seed=42

# 进程与节点配置
NPROC=$(echo "${gpu_ids}" | tr ',' '\n' | wc -l)
NNODES=${SLURM_NNODES}

# 环境变量
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_ids}

# 多机多卡启动
torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${NPROC} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  roboverse_learn/algorithms/diffusion_policy/train.py \
    --config-name=${config_name}.yaml \
    task.name=${task_name}_obs:${obs_space}_act:${act_space} \
    task.dataset.zarr_path="data_policy/${task_name}_obs:${obs_space}_act:${act_space}_${expert_data_num}.zarr" \
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
    ++policy.obs_encoder.test_rescale=${test_rescale} \
    ++task.dataset.max_visible_ratio=${max_visible_ratio} \
    ++optimizer.multigpu_lr_policy=${multigpu_lr_policy}
