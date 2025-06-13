export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
task_name=CloseBox
level=0
config_name=robot_dp
num_epochs=200
port=50061
norm_pnt_cloud=0
seed=42
gpus=4,5,6,7
strategy=linear
train_ratio=100
logger=wandb
consistent_warmup=0
bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_isaaclab/"${task_name}"-Level"${level}"/robot-franka "${task_name}"FrankaL"${level}" 100 "${gpus}" "${num_epochs}" joint_pos joint_pos 0 1 1 "${config_name}" "${port}" ${train_ratio} ${strategy} ${seed} ${norm_pnt_cloud} ${logger} ${consistent_warmup}
