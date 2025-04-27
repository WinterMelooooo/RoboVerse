import argparse
import json
import logging
import os
import shutil

import imageio.v2 as iio
import numpy as np
import torch
import zarr
from tqdm import tqdm

from roboverse_learn.algorithms.utils.pnt_cloud_getter import PntCloudGetter

try:
    from pytorch3d import transforms
except ImportError:
    pass

temp_dict = {
    "cam_pos": [1.5, 0.0, 1.5],
    "cam_look_at": [0.0, 0.0, 0.0],
    "cam_intr": [[293.19970703125, 0.0, 128.0], [0.0, 293.19970703125, 128.0], [0.0, 0.0, 1.0]],
    "cam_extr": [
        [0.0, 1.0, -0.0, -0.0],
        [0.7071067690849304, -0.0, -0.7071067690849304, -0.0],
        [-0.7071068286895752, 0.0, -0.7071068286895752, 2.1213204860687256],
        [0.0, 0.0, 0.0, 1.0],
    ],
}


demo_dir = r"/home/ghr/yktang/RoboVerse/roboverse_demo/demo_isaaclab/StackCube-Level0/robot-franka/demo_0001"
rgbs = iio.mimread(os.path.join(demo_dir, "rgb.mp4"))
depths = iio.mimread(os.path.join(demo_dir, "depth_uint8.mp4"))
with open(os.path.join(demo_dir, "metadata.json"), encoding="utf-8") as f:
    metadata = json.load(f)

pnt_cloud_getter = PntCloudGetter("StackCube", use_point_crop=True)


i = 0
rgb = rgbs[i]  # (256,256,3) [0,255]
depth = depths[i][:, :, 0] / 255.0  # (256,256) [0,1]
# print(max(depth.flatten()), min(depth.flatten()), depth.shape, type(depth[0, 0]))
cam_intr = np.array(metadata["cam_intr"][i])
cam_extr = np.array(metadata["cam_extr"][i])
depth_min = metadata["depth_min"][i]
depth_max = metadata["depth_max"][i]
cam_intr = np.array(temp_dict["cam_intr"]) if not cam_intr.size else cam_intr
cam_extr = np.array(temp_dict["cam_extr"]) if not cam_extr.size else cam_extr
# depth_meter = depth_min / (1 - depth * (1 - depth_min / depth_max)) # Use this for mujoco
depth_meter = depth_min + (depth.astype(np.float32)) * (depth_max - depth_min)
pnt_cloud = pnt_cloud_getter.get_point_cloud(
    rgb, np.ascontiguousarray(depth_meter).astype(np.float32), cam_intr, cam_extr
)
from roboverse_learn.algorithms.utils.visualizer import visualizer

visualizer.visualize_pointcloud(pnt_cloud)
