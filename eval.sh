export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=2
python roboverse_learn/eval.py --task CloseBox --algo diffusion_policy --max_step 500 --num_envs 50 --task_id_range_low 50 --task_id_range_high 100 --random.level 0 --headless --checkpoint_path /home/ghr/yktang/RoboVerse/info/outputs/DP/2025.05.11/16.24.42_CloseBoxFrankaL0_obs:joint_pos_act:joint_pos/checkpoints/1500.ckpt
#python roboverse_learn/eval.py --task StackCube --algo diffusion_policy --max_step 500 --num_envs 50 --task_id_range_low 0 --task_id_range_high 100 --random.level 0 --headless --checkpoint_path /home/ghr/yktang/RoboVerse/info/outputs/DP/2025.05.10/19.32.51_rgbd_vit_StackCubeFrankaL0_obs:joint_pos_act:joint_pos/checkpoints/2000.ckpt
