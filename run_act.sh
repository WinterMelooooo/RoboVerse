export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
task_name=CloseBox
level=0
#config_name=robot_dp_test_rgbd
num_epochs=2000
port=50049
seed=42
gpus=4,5,6,7
#train_ratio=100
bash roboverse_learn/algorithms/act/train_act.sh roboverse_demo/demo_isaaclab/"${task_name}"-Level"${level}"/robot-franka "${task_name}"FrankaL"${level}" 100 "${gpus}" "${num_epochs}" joint_pos joint_pos ${port}
