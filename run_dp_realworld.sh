export WANDB_API_KEY=263d7ff027d2057a793fb7f51783d43f5b6344cc
task_name=RealworldPickBottle
config_name=robot_dp_pntcloud_spUnet
num_epochs=3000
port=50372
seed=42
gpus=4,5,6,7
strategy=sqrt
train_ratio=100
logger=wandb
consistent_warmup=1
backend=Gloo
use_ee=0
obs_space=joint_pos
act_space=joint_pos
if [ "${use_ee}" -eq 1 ]; then
  obs_space=ee
  act_space=ee
fi

bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_realworld/"${task_name}"/robot-franka "${task_name}"FrankaRealWorld 150 "${gpus}" "${num_epochs}" "${obs_space}" "${act_space}" 0 1 1 "${config_name}" "${port}" ${train_ratio} ${strategy} ${seed} ${logger} ${consistent_warmup} ${backend}
