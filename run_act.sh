export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
task_name=CloseBox
level=2
config_name=robot_act_rgb
gpus=5
port=50053
num_epochs=2000
seed=42
expert_data_num=100
bash roboverse_learn/algorithms/act/train_act.sh roboverse_demo/demo_isaaclab/"${task_name}"-Level"${level}"/robot-franka "${task_name}"FrankaL"${level}" ${expert_data_num} "${gpus}" "${num_epochs}" joint_pos joint_pos 0 ${config_name} ${port} ${seed}
