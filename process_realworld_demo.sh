export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=1
task_name=RealworldLibero
use_ee=0

ee_flag=""
action_space="joint_pos"
observation_space="joint_pos"
if [ ${use_ee} -eq 1 ]; then
    ee_flag="--use_ee"
    action_space="ee"
    observation_space="ee"
fi




bash roboverse_learn/algorithms/diffusion_policy/data2zarr_realworld.sh roboverse_demo/demo_realworld/${task_name}/robot-franka ${task_name}FrankaRealWorld 100 ${observation_space} ${action_space} 0 1 1
