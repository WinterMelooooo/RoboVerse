import numpy as np
import torch

from roboverse_learn.algorithms.utils.visualizer import visualizer

idx = np.random.randint(0, 10000)
pnt_cloud = np.load(f"/home/balen/Projects/yktang/RoboVerse/tmp/visualize/CloseBoxL0/demo_0000_step_0.npy")
# print(idx)
visualizer.visualize_pointcloud(pnt_cloud)
