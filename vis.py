import numpy as np
import torch

from roboverse_learn.algorithms.utils.visualizer import visualizer

idx = np.random.randint(0, 10000)
pnt_cloud = np.load(f"/home/ghr/yktang/RoboVerse/data_policy/RealworldLiberoPickButterFrankaRealWorld_obs:joint_pos_act:joint_pos_50.zarr/pnt_cloud.npy")
print(f"Point cloud shape: {pnt_cloud.shape}, dtype: {pnt_cloud.dtype}")
# print(idx)
visualizer.visualize_pointcloud(pnt_cloud[..., :6])
