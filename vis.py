import numpy as np
import torch

from roboverse_learn.algorithms.utils.visualizer import visualizer

idx = np.random.randint(0, 200)
pnt_cloud = np.load(f"/home/ghr/yktang/RoboVerse/tmp/visualize/StackCube/pnt_cloud/{idx}.npy")
print(idx)
visualizer.visualize_pointcloud(pnt_cloud)
