export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
task_name=ExtLibero100_KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet
config_name=robot_dp_resnet_pointnet_mutual_attention_dropout_ee
num_epochs=4000
port=50350
seed=42
gpus=0,1,2,3,4,5,6,7
strategy=sqrt
train_ratio=100
logger=wandb
consistent_warmup=1
backend=Gloo
bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_isaaclab/"${task_name}"/robot-franka "${task_name}"Franka 47 "${gpus}" "${num_epochs}" joint_pos joint_pos 0 1 1 "${config_name}" "${port}" ${train_ratio} ${strategy} ${seed} ${logger} ${consistent_warmup} ${backend}
