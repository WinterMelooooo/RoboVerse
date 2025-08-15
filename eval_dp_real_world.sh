export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=0
ckpt=/home/balen/Projects/yktang/RoboVerse/info/outputs/DP/2025.08.14/19.50.09_RGB_ResNet18_RealworldLiberoPickButterFrankaRealWorld_obs:joint_pos_act:joint_pos/checkpoints/2000.ckpt
task=LiberoPickButter
num_envs=1
#random_level=2
#use_touch=0
task_id_range_low=0
task_id_range_high=200

# touch=""
# if [ ${use_touch} -eq 1 ]; then
#     touch="--use_touch"
# fi

python roboverse_learn/eval_real_world.py --task ${task} --algo diffusion_policy --max_step 400 --num_envs ${num_envs} --task_id_range_low ${task_id_range_low} --task_id_range_high ${task_id_range_high}  --headless --checkpoint_path ${ckpt}
