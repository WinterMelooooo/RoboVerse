export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#python metasim/scripts/collect_demo.py --task=CloseBox --num_envs=50 --run_all --headless
bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka CloseBoxFrankaL0 100 2,4,5,6 2000 joint_pos joint_pos 0 1 1 robot_dp_rgbd_vit 0 50032
#python metasim/scripts/collect_demo.py --task=StackCube --num_envs=50 --run_all --headless
#bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_isaaclab/StackCube-Level0/robot-franka StackCubeFrankaL0 100 0,1,4,5 300 joint_pos joint_pos 0 1 1 robot_dp_test_rgbd 0 50031
