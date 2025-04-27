import numpy as np
import torch

from roboverse_learn.algorithms.utils.visualizer import visualizer

idx = np.random.randint(0, 17923)
pnt_cloud = np.load(f"/home/ghr/yktang/RoboVerse/tmp/pnt_cloud/dataset_pntcloud/pntcloud_dataset_{idx}.npy")
print(idx)
visualizer.visualize_pointcloud(pnt_cloud)
