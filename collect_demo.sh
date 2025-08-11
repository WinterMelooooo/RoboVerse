export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=7
task_name=LiberoPickButter
random_level=4
num_envs=1
demo_start_idx=0
max_demo_idx=1000
target_num_demos=500
use_touch=1
touch_flag=$([ "$use_touch" -eq 1 ] && echo "--use_touch")

#python metasim/scripts/collect_demo.py --task=${task_name} --num_envs=${num_envs} --run_all --headless --random.level=${random_level} --demo_start_idx=${demo_start_idx} --max_demo_idx=${max_demo_idx} --target_num_demos=${target_num_demos} ${touch_flag}
bash roboverse_learn/algorithms/diffusion_policy/data2zarr.sh roboverse_demo/demo_isaaclab/${task_name}-Level${random_level}/robot-franka ${task_name}FrankaL${random_level} ${target_num_demos} joint_pos joint_pos 0 1 1
