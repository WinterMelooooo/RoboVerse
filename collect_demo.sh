export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=1
#python metasim/scripts/collect_demo.py --task=CloseBox --num_envs=1 --run_all --headless --random.level 2
#bash roboverse_learn/algorithms/diffusion_policy/data2zarr.sh roboverse_demo/demo_isaaclab/CloseBox-Level2/robot-franka CloseBoxFrankaL2 100 2,3,6,7 400 joint_pos joint_pos 0 1 1
#python metasim/scripts/collect_demo.py --task=StackCube --num_envs=1 --run_all --headless --random.level 2
bash roboverse_learn/algorithms/diffusion_policy/data2zarr.sh roboverse_demo/demo_isaaclab/StackCube-Level2/robot-franka StackCubeFrankaL2 100 2,3,6,7 400 joint_pos joint_pos 0 1 1
#bash roboverse_learn/algorithms/diffusion_policy/data2zarr.sh roboverse_demo/demo_isaaclab/StackCube-Level0/robot-franka StackCubeFrankaL0_len1000 1000 2,3,6,7 400 joint_pos joint_pos 0 1 1 0
