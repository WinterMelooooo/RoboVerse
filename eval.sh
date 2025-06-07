export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=5
ckpt=/home/ghr/yktang/RoboVerse/info/outputs/DP/2025.06.07/05.46.09_PntCloud_DP3_CloseBoxFrankaL2_obs:joint_pos_act:joint_pos/checkpoints/2000.ckpt
task=CloseBox
num_envs=1
random_level=2
task_id_range_low=0
task_id_range_high=200
python roboverse_learn/eval.py --task ${task} --algo diffusion_policy --max_step 500 --num_envs ${num_envs} --task_id_range_low ${task_id_range_low} --task_id_range_high ${task_id_range_high} --random.level ${random_level} --headless --checkpoint_path ${ckpt}
#python roboverse_learn/eval.py --task StackCube --algo diffusion_policy --max_step 500 --num_envs 50 --task_id_range_low 0 --task_id_range_high 100 --random.level 0 --headless --checkpoint_path /home/ghr/yktang/RoboVerse/info/outputs/DP/2025.05.10/19.32.51_rgbd_vit_StackCubeFrankaL0_obs:joint_pos_act:joint_pos/checkpoints/2000.ckpt
