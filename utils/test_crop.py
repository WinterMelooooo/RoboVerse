import argparse
import json
import logging
import os
import shutil
import sys
import time

import imageio.v2 as iio
import numpy as np
import torch
import zarr
from tqdm import tqdm

sys.path.append(".")
from roboverse_learn.algorithms.utils.img_processing import _center_crop_and_resize

try:
    from pytorch3d import transforms
except ImportError:
    pass
rgbs = iio.mimread(
    "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_realworld/LiberoPickButter/robot-franka/demo_0000/rgb.mp4",
    memtest=False,
)
rgb = rgbs[0]
rgb = _center_crop_and_resize(rgb, target_width=256, target_height=256)
print(f"Time taken to crop and resize: {end - start:.4f} seconds")
print(f"Shape of cropped and resized image: {rgb.shape}")
iio.imwrite(
    "/home/ghr/yktang/RoboVerse/test_crop.png",
    rgb,
)
