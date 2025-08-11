import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # ✅ 新增：用于拼图

# 数据集路径
root_dir = "roboverse_demo/demo_realworld/RealworldLiberoPickButter-complex/robot-franka"
# root_dir = "roboverse_demo/demo_isaaclab/LiberoPickButter-Level0/robot-franka"
demo_name = root_dir.split("/")[-2]
if "-" in demo_name:
    demo_name = demo_name.split("-")[0]

# 找到所有 demo 目录
demo_dirs = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if d.startswith("demo_")])

# 初始化关节数据列表
joint_data = [[] for _ in range(9)]  # 9 个关节，每个元素是一个 list，存不同 demo 的曲线

for demo_dir in demo_dirs:
    metadata_path = os.path.join(demo_dir, "metadata.pkl")
    if not os.path.exists(metadata_path):
        print(f"跳过 {demo_dir}（无 metadata.pkl）")
        continue

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    if "joint_qpos" not in metadata:
        print(f"跳过 {demo_dir}（无 joint_qpos）")
        continue

    qpos = np.array(metadata["joint_qpos"])  # [T, 9]
    for joint_idx in range(9):
        joint_data[joint_idx].append(qpos[:, joint_idx])

# 绘图保存
save_dir = os.path.join("./tmp/joint_qpos_plots", demo_name)
os.makedirs(save_dir, exist_ok=True)

def get_range_with_padding(values, pad_ratio=0.05):
    """给当前关节所有 demo 的拼接数据加一点边距，防止贴边"""
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        # 全程几乎不动时，给个很小的可视范围
        delta = 1e-3
        return vmin - delta, vmax + delta
    pad = (vmax - vmin) * pad_ratio
    return vmin - pad, vmax + pad

# 逐关节画图
for joint_idx in range(9):
    if len(joint_data[joint_idx]) == 0:
        print(f"关节 {joint_idx} 无数据，跳过")
        continue

    # 将该关节的所有 demo 串起来用于范围统计
    all_vals = np.concatenate(joint_data[joint_idx], axis=0)
    ymin, ymax = get_range_with_padding(all_vals, pad_ratio=0.05)

    plt.figure(figsize=(10, 6))
    for demo_curve in joint_data[joint_idx]:
        plt.plot(range(len(demo_curve)), demo_curve, alpha=0.6)
    plt.xlabel("Timestep")
    plt.ylabel(f"Joint {joint_idx} Position (rad)")  # 如果是角度自己改成 deg
    plt.title(f"Joint {joint_idx} Position Change (All Demos)")
    plt.ylim(ymin, ymax)      # ✅ 每个关节使用自己的 y 轴范围
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"joint_{joint_idx}.png"))
    plt.close()

print(f"✅ 绘制完成，图片保存在 {save_dir}/，每个关节使用独立纵轴范围。")

# ======= 新增：把最后 9 张图拼成 3×3 大图 =======
grid_rows, grid_cols = 3, 3
paths = [os.path.join(save_dir, f"joint_{i}.png") for i in range(9)]

# 读取第一张存在的图，确定单元格尺寸
first_img = None
for p in paths:
    if os.path.exists(p):
        first_img = Image.open(p).convert("RGB")
        break

# 若9张都不存在，直接退出
if first_img is None:
    print("未找到任何关节图片，跳过拼图。")
else:
    cell_w, cell_h = first_img.size

    # 若存在图片大小不一致，统一 resize 到第一张的尺寸
    imgs = []
    for p in paths:
        if os.path.exists(p):
            im = Image.open(p).convert("RGB")
            if im.size != (cell_w, cell_h):
                im = im.resize((cell_w, cell_h))
            imgs.append(im)
        else:
            # 用白色占位
            placeholder = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
            imgs.append(placeholder)

    grid_w = grid_cols * cell_w
    grid_h = grid_rows * cell_h
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    # 依次贴到 3×3 网格
    for idx, im in enumerate(imgs):
        r = idx // grid_cols
        c = idx % grid_cols
        grid.paste(im, (c * cell_w, r * cell_h))

    grid_path = os.path.join(save_dir, "joint_grid.png")
    grid.save(grid_path)
    print(f"🧩 已生成 3×3 拼图：{grid_path}")
