export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=1
ckpt=info/outputs/DP/2025.09.14/19.49.54_RGBD_ViT_LiberoPickAlphabetSoupFrankaL1_obs:joint_pos_act:joint_pos/checkpoints/best.ckpt
task=LiberoPickAlphabetSoup
num_envs=1
random_level=0
use_touch=0
task_id_range_low=0
task_id_range_high=200

touch=""
if [ ${use_touch} -eq 1 ]; then
    touch="--use_touch"
fi

python roboverse_learn/eval.py --task ${task} --algo diffusion_policy --max_step 250 --num_envs ${num_envs} --task_id_range_low ${task_id_range_low} --task_id_range_high ${task_id_range_high} --random.level ${random_level} --headless --checkpoint_path ${ckpt} ${touch}
