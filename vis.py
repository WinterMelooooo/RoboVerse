import numpy as np
import torch

from roboverse_learn.algorithms.utils.visualizer import visualizer

idx = np.random.randint(0, 50)
idx = 0
pnt_cloud = np.load(f"tmp/visualize/eval/pntcloud_eval_{idx}.npy")
print(idx)
visualizer.visualize_pointcloud(pnt_cloud)
