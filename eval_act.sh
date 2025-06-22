export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=6
ckpt=./info/outputs/ACT/2025.06.22/04.46.18_RGB_ResNet18_CloseBoxFrankaL2_obs:joint_pos_act:joint_pos/checkpoints/best.ckpt
task=CloseBox
num_envs=1
random_level=2
task_id_range_low=0
task_id_range_high=200
python roboverse_learn/eval.py --task ${task} --algo ACT --max_step 500 --num_envs ${num_envs} --task_id_range_low ${task_id_range_low} --task_id_range_high ${task_id_range_high} --random.level ${random_level} --headless --checkpoint_path ${ckpt}
