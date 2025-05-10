export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=7
python roboverse_learn/eval.py --task CloseBox --algo diffusion_policy --max_step 500 --num_envs 50 --task_id_range_low 0 --task_id_range_high 100 --random.level 0 --headless --checkpoint_path /home/ghr/yktang/RoboVerse/info/outputs/DP/2025.05.03/18.17.46_CloseBoxFrankaL0_obs:joint_pos_act:joint_pos/checkpoints/2000.ckpt
#python roboverse_learn/eval.py --task StackCube --algo diffusion_policy --max_step 500 --num_envs 50 --task_id_range_low 0 --task_id_range_high 200 --random.level 0 --headless --checkpoint_path /home/ghr/yktang/RoboVerse/info/outputs/DP/2025.05.08/08.02.20_pntcloud_dp3_StackCubeFrankaL0_obs:joint_pos_act:joint_pos/checkpoints/2000.ckpt
