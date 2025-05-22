export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=2
task_name=SlideCabinetOpenAndPlaceCups
random_level=2
num_envs=1
demo_start_idx=0
max_demo_idx=300
python metasim/scripts/collect_demo.py --task=${task_name} --num_envs=${num_envs} --run_all --headless --random.level=${random_level} --demo_start_idx=${demo_start_idx} --max_demo_idx=${max_demo_idx}
bash roboverse_learn/algorithms/diffusion_policy/data2zarr.sh roboverse_demo/demo_isaaclab/${task_name}-Level${random_level}/robot-franka ${task_name}FrankaL${random_level} 100 2,3,6,7 400 joint_pos joint_pos 0 1 1
