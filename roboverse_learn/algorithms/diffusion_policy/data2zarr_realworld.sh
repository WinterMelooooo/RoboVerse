# Examples:
# bash roboverse_learn/algorithms/diffusion_policy/train_dp.sh roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka CloseBoxFrankaL0 100 0 200 joint_pos joint_pos

# 'metadata_dir' means path to metadata directory. e.g. roboverse_demo/demo_isaaclab/CloseBox-Level0/robot-franka
# 'task_name' gives a name to the policy, which can include the task robot and level ie CloseBoxFrankaL0
# 'expert_data_num' means number of training data. e.g.100
# 'gpu_ids' means single gpu id, e.g.0

metadata_dir=${1}
task_name=${2}
expert_data_num=${3}
obs_space=${4} # joint_pos or ee
act_space=${5} # joint_pos or ee
delta_ee=${6:-0} # 0 or 1 (only matters if act_space is ee, 0 means absolute 1 means delta control )
store_rgbd=${7:-0} # 0 or 1
store_pnt_cloud=${8:-0} # 0 or 1


# adding the obs and action space as additional info
extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

python roboverse_learn/algorithms/data2zarr_dp_realworld.py \
--task_name ${task_name}_${extra} \
--expert_data_num ${expert_data_num} \
--metadata_dir ${metadata_dir} \
--action_space ${act_space} \
--observation_space ${obs_space} \
--delta_ee ${delta_ee} \
--store_rgbd ${store_rgbd} \
--store_pnt_cloud ${store_pnt_cloud} \
