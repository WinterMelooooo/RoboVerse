import os
import sys
import time
import numpy as np
import cv2
from PIL import Image
def _center_crop_and_resize( img: Image.Image, target_width: int, target_height: int) -> Image.Image:

    orig_w, orig_h = img.size
    target_ratio = target_width / target_height
    orig_ratio   = orig_w / orig_h

    if orig_ratio > target_ratio:
        new_h = orig_h
        new_w = int(target_ratio * new_h)
    else:
        new_w = orig_w
        new_h = int(new_w / target_ratio)

    # 计算中心裁剪区域
    left   = (orig_w - new_w) // 2
    top    = (orig_h - new_h) // 2
    right  = left + new_w
    bottom = top  + new_h
    img_cropped = img.crop((left, top, right, bottom))
    img_cropped = img_cropped.resize((target_width, target_height), Image.LANCZOS)
    return np.array(img_cropped)


img = Image.open("/home/user/yktang/RoboVerse/roboverse_demo/demo-realworld/RealworldLiberoPickButter/robot-franka/demo_0010/demo_rgb.png")
img_resized = _center_crop_and_resize(img, 256, 256)

import matplotlib.pyplot as plt

plt.imshow(img_resized)
plt.axis('off')
plt.show()
