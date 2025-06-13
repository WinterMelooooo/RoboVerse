export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=4
ckpt=/home/ghr/yktang/RoboVerse/info/outputs/DP/2025.06.12/13.03.09_Fusion_ResNet18_PointNet_Joint_Attention_StackCubeFrankaL0_obs:joint_pos_act:joint_pos/checkpoints/2000.ckpt
task=StackCube
num_envs=50
random_level=0
task_id_range_low=0
task_id_range_high=200
use_segmentation_mask=0

mask_arg=""
if [ "${use_segmentation_mask}" -eq 1 ]; then
  mask_arg="--use_segmentation_mask"
fi

python roboverse_learn/eval.py --task ${task} --algo diffusion_policy --max_step 500 --num_envs ${num_envs} --task_id_range_low ${task_id_range_low} --task_id_range_high ${task_id_range_high} --random.level ${random_level} --headless --checkpoint_path ${ckpt} ${mask_arg}
