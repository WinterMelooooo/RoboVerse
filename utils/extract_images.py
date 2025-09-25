import imageio as iio
import os
import time
from rembg import remove
from PIL import Image
from tqdm import tqdm
os.environ["U2NET_DEVICE"] = "cuda"

src = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_realworld/RealworldPour/robot-franka"
dest = "/home/ghr/yktang/RoboVerse/tmp/demo_imgs/RealworldPour"
os.makedirs(dest, exist_ok=True)


def extract_images_from_video(demo_dir, dest_dir, step = 5):
    rgb_path = os.path.join(demo_dir, "rgb.mp4")
    rgbs = iio.get_reader(rgb_path)
    os.makedirs(dest_dir, exist_ok=True)
    for frame_idx, rgb in enumerate(rgbs):
        if frame_idx % step == 0:
            rgb_name = f"color_image_{frame_idx//step:04d}.png"
            iio.imwrite(os.path.join(dest_dir, rgb_name), rgb)
    print(f"Extracted images from {demo_dir} to {dest_dir}")
    return


def main():
    demo_id_list =
