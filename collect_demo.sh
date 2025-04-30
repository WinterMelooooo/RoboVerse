export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export CUDA_VISIBLE_DEVICES=7
#python metasim/scripts/collect_demo.py --task=CloseBox --num_envs=1 --run_all --headless --random.level 1
#bash roboverse_learn/algorithms/diffusion_policy/data2zarr.sh roboverse_demo/demo_isaaclab/CloseBox-Level1/robot-franka CloseBoxFrankaL1 100 2,3,6,7 400 joint_pos joint_pos 0 1 1
python metasim/scripts/collect_demo.py --task=StackCube --num_envs=1 --run_all --headless --random.level 1
bash roboverse_learn/algorithms/diffusion_policy/data2zarr.sh roboverse_demo/demo_isaaclab/StackCube-Level1/robot-franka StackCubeFrankaL1 100 2,3,6,7 400 joint_pos joint_pos 0 1 1
