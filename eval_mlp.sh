export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=1
ckpt=info/outputs/MLP/2025.07.09/08.05.51_RGB_ResNet18_CloseBoxFrankaL2_obs:joint_pos_act:joint_pos/checkpoints/200.ckpt
task=CloseBox
num_envs=1
random_level=2
use_touch=0
task_id_range_low=0
task_id_range_high=200

touch=""
if [ ${use_touch} -eq 1 ]; then
    touch="--use_touch"
fi

python roboverse_learn/eval.py --task ${task} --algo mlp --max_step 500 --num_envs ${num_envs} --task_id_range_low ${task_id_range_low} --task_id_range_high ${task_id_range_high} --random.level ${random_level} --headless --checkpoint_path ${ckpt} ${touch}
