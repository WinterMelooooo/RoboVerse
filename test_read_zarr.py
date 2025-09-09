import zarr

from roboverse_learn.algorithms.utils.visualizer import visualizer

zarr_path = "/home/ghr/yktang/RoboVerse/data_policy/StackCubeFrankaL0_obs:joint_pos_act:joint_pos_100.zarr"
root = zarr.open(zarr_path, mode="r")

data = root["data/head_camera_pnt_cloud"][:]
print(data.shape)
visualizer.visualize_pointcloud(data[..., :6][50])
