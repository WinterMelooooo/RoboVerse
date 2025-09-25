export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=1
export HEADLESS=1
export PYGLFW_LIBRARY=$CONDA_PREFIX/lib/libglfw.so.3
ckpt=./info/outputs/DP/2025.09.15/18.43.13_Fusion_ResNet18_PointNet_Mutual_Attention_Dropout_ExtLibero100_KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinetFranka_obs:joint_pos_act:joint_pos/checkpoints/4000.ckpt

# The following is not used, but kept for interface consistency
task=LiberoPickButter
num_envs=1
random_level=4
use_touch=0
task_id_range_low=0
task_id_range_high=200

touch=""
if [ ${use_touch} -eq 1 ]; then
    touch="--use_touch"
fi

python third_party/experiments/robot/libero/run_libero_eval.py --task ${task} --algo diffusion_policy --max_step 250 --num_envs ${num_envs} --task_id_range_low ${task_id_range_low} --task_id_range_high ${task_id_range_high} --random.level ${random_level} --headless --checkpoint_path ${ckpt} ${touch}
