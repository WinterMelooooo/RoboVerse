export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=3
task_name=Ext_Libero_100
num_envs=1
demo_start_idx=0
max_demo_idx=3000
target_num_demos=95
use_touch=0
use_random_traj=0
touch_flag=$([ "$use_touch" -eq 1 ] && echo "--use_touch")
rand_traj_flag=$([ "$use_random_traj" -eq 1 ] && echo "--use_random_traj")

#python metasim/scripts/collect_demo.py --task=${task_name} --num_envs=${num_envs} --run_all --headless --random.level=${random_level} --demo_start_idx=${demo_start_idx} --max_demo_idx=${max_demo_idx} --target_num_demos=${target_num_demos} ${touch_flag} ${rand_traj_flag}
bash roboverse_learn/algorithms/diffusion_policy/data2zarr.sh roboverse_demo/demo_libero/${task_name}/robot-franka ${task_name}Franka ${target_num_demos} joint_pos joint_pos 0 1 1
