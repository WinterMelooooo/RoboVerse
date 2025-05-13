export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_isaaclab/StackCube-Level0/robot-franka StackCubeFrankaL0 100 1,2,3,4,5,6,7 2000 joint_pos joint_pos 0 1 1 robot_dp_pntcloud_spUnet 1 50048 50 sqrt
#bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_isaaclab/StackCube-Level0/robot-franka StackCubeFrankaL0 100 4,5,6,7 2000 joint_pos joint_pos 0 1 1 robot_dp_pntcloud 1 50036 100 sqrt
#bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_isaaclab/StackCube-Level/robot-franka StackCubeFrankaL0 100 2,3,4,5 2000 joint_pos joint_pos 0 1 1 robot_dp_pntcloud_spUnet 1 50043 100 linear 16 2 8
