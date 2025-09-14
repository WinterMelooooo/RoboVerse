import numpy as np
import torch

from roboverse_learn.algorithms.utils.visualizer import visualizer

idx = np.random.randint(0, 10000)
pnt_cloud = np.load(f"/home/ghr/yktang/RoboVerse/roboverse_demo/demo_libero/ExtLibero100/robot-franka/demo_0000/point_clouds.npy")
print(f"Point cloud shape: {pnt_cloud.shape}, dtype: {pnt_cloud.dtype}")
# print(idx)
# print(f"Point cloud shape: {pnt_cloud.shape}, dtype: {pnt_cloud.dtype}")
visualizer.visualize_pointcloud(pnt_cloud[..., :6][0])
