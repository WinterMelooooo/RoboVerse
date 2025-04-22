export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
#CUDA_VISIBLE_DEVICES=0
python roboverse_learn/eval.py --task CloseBox --algo diffusion_policy --max_step 1000 --num_envs 100 --headless --checkpoint_path /home/ghr/yktang/RoboVerse/info/outputs/DP/2025.04.21/07.07.48_CloseBoxFrankaL0_obs:joint_pos_act:joint_pos/checkpoints/150.ckpt
