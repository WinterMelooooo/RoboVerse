export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
export CUDA_VISIBLE_DEVICES=6
python roboverse_learn/eval.py --task CloseBox --algo diffusion_policy --max_step 500 --num_envs 50 --max_demo 100 --random.level 0 --headless --checkpoint_path /home/ghr/yktang/RoboVerse/info/outputs/DP/2025.04.28/07.01.09_CloseBoxFrankaL0_obs:joint_pos_act:joint_pos/checkpoints/1800.ckpt
#python roboverse_learn/eval.py --task StackCube --algo diffusion_policy --max_step 500 --num_envs 50 --max_demo 100 --random.level 0 --headless --checkpoint_path /home/ghr/yktang/RoboVerse/info/outputs/DP/2025.04.29/08.04.38_StackCubeFrankaL0_obs:joint_pos_act:joint_pos/checkpoints/300.ckpt
