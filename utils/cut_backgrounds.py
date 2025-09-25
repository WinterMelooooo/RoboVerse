import imageio as iio
import os
import time
from rembg import remove
from PIL import Image
from tqdm import tqdm
os.environ["U2NET_DEVICE"] = "cuda"

src = "/home/ghr/yktang/RoboVerse/tmp/demo_imgs/Realworld_Pour"
dest = "/home/ghr/yktang/RoboVerse/tmp/demo_imgs/Realworld_Pour"
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


def cut_background(image_path: str, dest_dir: str, out_name: str = None) -> str:
    """
    将输入图片的背景分割掉并保存为4通道(RGBA)图片。

    参数
    ----
    image_path : str
        输入图片路径。
    dest_dir : str
        结果保存目录。
    out_name : str, 可选
        输出文件名（不含路径）。若不提供，则使用原文件名并强制后缀为.png。

    返回
    ----
    str : 输出图片的完整路径
    """
    # 1. 确保输出目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 2. 打开输入图片
    input_img = Image.open(image_path).convert("RGBA")

    # 3. 调用 rembg 去背景
    output_img = remove(input_img)   # 返回 Pillow Image，已是 RGBA

    # 4. 生成输出文件名
    if out_name is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_name = f"{base}.png"     # 统一保存为 PNG，保留 alpha 通道

    out_path = os.path.join(dest_dir, out_name)

    # 5. 保存到目标目录
    output_img.save(out_path, format="PNG")
    print(f"Saved cutout image to {out_path}")
    return out_path


imgs = sorted(os.listdir(src))
for img in tqdm(imgs):
    img_path = os.path.join(src, img)
    cut_background(img_path, dest)
