#!/usr/bin/env bash
# run_slurm.sh — 提交两节点、每节点8卡的 Slurm 作业

#SBATCH --job-name=dp_multi_job
#SBATCH --partition=savio4_gpu
#SBATCH --account=co_rllab
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodelist=n0142.savio4,n0143.savio4


# rendezvous 设置
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=50023
export SLURM_NNODES=$SLURM_NNODES

# 指定每节点使用的 GPU 列表（0–7）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 调用多机脚本
export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
bash roboverse_learn/algorithms/diffusion_policy/train_dp_slurm.sh roboverse_demo/demo_isaaclab/CloseBox-Level1/robot-franka CloseBoxFrankaL2 100 0,1,2,3,4,5,6,7 2000 joint_pos joint_pos 0 1 1 robot_dp_rgbd_vit 1 50048 50 sqrt
