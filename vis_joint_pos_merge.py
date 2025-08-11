import os
from PIL import Image

# 输入两个文件夹路径
dir1 = "tmp/joint_qpos_plots/LiberoPickButter"
dir2 = "tmp/joint_qpos_plots/RealworldLiberoPickButter"

# 输出文件夹
save_dir = "tmp/joint_qpos_plots/merge"
os.makedirs(save_dir, exist_ok=True)

# 获取两个文件夹中的文件名并排序
files1 = sorted([f for f in os.listdir(dir1) if f.endswith(".png")])
files2 = sorted([f for f in os.listdir(dir2) if f.endswith(".png")])

# 只处理两个文件夹中都有的文件
common_files = sorted(list(set(files1) & set(files2)))

for fname in common_files:
    path1 = os.path.join(dir1, fname)
    path2 = os.path.join(dir2, fname)

    img1 = Image.open(path1)
    img2 = Image.open(path2)

    # 横向拼接
    new_width = img1.width + img2.width
    new_height = max(img1.height, img2.height)

    new_img = Image.new("RGB", (new_width, new_height), (255, 255, 255))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))

    save_path = os.path.join(save_dir, fname)
    new_img.save(save_path)

print(f"✅ 拼接完成，结果保存在 {save_dir}/")
