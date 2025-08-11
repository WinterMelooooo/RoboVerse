export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
task_name=LiberoPickButter
level=4
config_name=robot_dp_resnet_pointnet_mutual_attention_dropout
num_epochs=2000
port=50348
seed=42
gpus=0,1,3,4
strategy=sqrt
train_ratio=100
logger=wandb
consistent_warmup=1
backend=Gloo
bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_isaaclab/"${task_name}"-Level"${level}"/robot-franka "${task_name}"FrankaL"${level}" 100 "${gpus}" "${num_epochs}" joint_pos joint_pos 0 1 1 "${config_name}" "${port}" ${train_ratio} ${strategy} ${seed} ${logger} ${consistent_warmup} ${backend}
