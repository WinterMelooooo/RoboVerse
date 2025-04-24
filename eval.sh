export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
#CUDA_VISIBLE_DEVICES=4
python roboverse_learn/eval.py --task CloseBox --algo diffusion_policy --max_step 500 --num_envs 50 --headless --checkpoint_path /home/ghr/yktang/RoboVerse/info/outputs/DP/2025.04.20/04.53.50_CloseBoxFrankaL0_obs:joint_pos_act:joint_pos/checkpoints/200.ckpt
