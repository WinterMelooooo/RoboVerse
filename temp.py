import numpy as np

from roboverse_learn.algorithms.utils.visualizer import visualizer

loaded_pc = np.load("/home/ghr/yktang/RoboVerse/tmp/pnt_cloud/my_pointcloud_train.npy")
print(loaded_pc[0].shape)
visualizer.visualize_pointcloud(loaded_pc[0])
