export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=5
python roboverse_learn/eval.py --task CloseBox --algo diffusion_policy --max_step 500 --num_envs 1 --task_id_range_low 0 --task_id_range_high 100 --random.level 1 --headless --checkpoint_path /home/ghr/yktang/RoboVerse/info/outputs/DP/2025.05.06/01.51.03_CloseBoxFrankaL1_obs:joint_pos_act:joint_pos/checkpoints/2000.ckpt
#python roboverse_learn/eval.py --task StackCube --algo diffusion_policy --max_step 500 --num_envs 1 --task_id_range_low 0 --task_id_range_high 200 --random.level 1 --headless --checkpoint_path /home/ghr/yktang/RoboVerse/info/outputs/DP/2025.05.06/20.36.20_rgb_resnet18_StackCubeFrankaL1_obs:joint_pos_act:joint_pos/checkpoints/200.ckpt
