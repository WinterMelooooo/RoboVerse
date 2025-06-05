export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=1
ckpt=/home/ghr/yktang/RoboVerse/info/outputs/DP/2025.06.04/19.17.50_Fusion_ResNet18_PointNet_Late_Fusion_CloseBoxFrankaL0_obs:joint_pos_act:joint_pos/checkpoints/2000.ckpt
task=CloseBox
num_envs=50
random_level=0
python roboverse_learn/eval.py --task ${task} --algo diffusion_policy --max_step 500 --num_envs ${num_envs} --task_id_range_low 0 --task_id_range_high 100 --random.level ${random_level} --headless --checkpoint_path ${ckpt}
#python roboverse_learn/eval.py --task StackCube --algo diffusion_policy --max_step 500 --num_envs 50 --task_id_range_low 0 --task_id_range_high 100 --random.level 0 --headless --checkpoint_path /home/ghr/yktang/RoboVerse/info/outputs/DP/2025.05.10/19.32.51_rgbd_vit_StackCubeFrankaL0_obs:joint_pos_act:joint_pos/checkpoints/2000.ckpt
