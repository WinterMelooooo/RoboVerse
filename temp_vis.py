import glob
import os

import numpy as np
import torch

from roboverse_learn.algorithms.utils.visualizer import visualizer

# 1. 指定保存 pcd_dict 的目录
file = r"/home/ghr/yktang/RoboVerse/tmp/visualize/test/pcd_transformed_cuda_450_0_0.npz"

# 2. 列出所有 .npz 文件，并选取其中一个，比如第一帧
npz_files = sorted(glob.glob(file))
if not npz_files:
    raise RuntimeError(f"No .npz files found in {file}")

# 你也可以根据文件名后缀选择特定的 i、idx，例如：
# npz_files = [f for f in npz_files if f.endswith("pcd_transformed_0_0.npz")]
idx = np.random.randint(0, len(npz_files))
file_to_load = npz_files[0]  # 取第一个
print(f"Loading pointcloud——{idx} from {file_to_load}")

# 3. 加载 .npz，里面包含多个数组：coord, grid_coord, feat, offset
data = np.load(file_to_load)

# 4. 提取坐标（N,3）和颜色特征（如果需要）
coords = data["coord"]  # (N, 3)
feat = data["feat"]  # (N, F) e.g. F=6 (rgb+xyz) 或其它
feat = torch.from_numpy(feat).float()
rgb = (feat[:, :3] + 1) / 2 * 255  # (N, 3)
rgb.to(torch.uint8)
print(f"rgb min: {rgb.min()}, max: {rgb.max()}")
xyz = feat[:, 3:6]  # (N, 3)
xyz_rgb = torch.cat([xyz, rgb], dim=1)  # (N, 6)
# 如果 feat 包含 color+coord，可以直接当六维点云可视化：
your_pointcloud = xyz_rgb.cpu().numpy()  # (N, 6)

# 或者只可视化坐标
# your_pointcloud = coords  # 或 np.concatenate([coords, feat], axis=1)

# 5. 调用可视化工具
visualizer.visualize_pointcloud(your_pointcloud)
