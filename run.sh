export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
task_name=CloseBox
level=2
config_name=robot_dp_resnet_pointnet_mutual_attention
num_epochs=2000
port=50095
norm_pnt_cloud=0
seed=42
gpus=0,1,2,3,4,5,6,7
strategy=sqrt
bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_isaaclab/"${task_name}"-Level"${level}"/robot-franka "${task_name}"FrankaL"${level}" 100 "${gpus}" "${num_epochs}" joint_pos joint_pos 0 1 1 "${config_name}" "${port}" 50 ${strategy} ${seed} ${norm_pnt_cloud}
