 export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=1
task_name=RealworldLiberoPickButter



bash roboverse_learn/algorithms/diffusion_policy/data2zarr_realworld.sh roboverse_demo/demo_realworld/${task_name}/robot-franka ${task_name}FrankaRealWorld 50 joint_pos joint_pos 0 1 1
