export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=7
ckpt=/home/ghr/yktang/RoboVerse/info/outputs/ACT/2025.06.17/01.16.03_CloseBoxFrankaL0_obs:joint_pos_act:joint_pos_100/policy_last.ckpt
task=CloseBox
num_envs=50
random_level=0
python roboverse_learn/eval.py --task ${task} --algo ACT --max_step 500 --num_envs ${num_envs} --task_id_range_low 0 --task_id_range_high 100 --random.level ${random_level} --headless --checkpoint_path ${ckpt}
