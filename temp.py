#!/usr/bin/env python3
import argparse
import os

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import zarr
from tqdm import tqdm

from roboverse_learn.algorithms.utils.visualizer import visualizer


def main():
    parser = argparse.ArgumentParser(
        description="从 ZARR 文件中随机抽取一个点云并可视化，保存为 .npy 文件"
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        required=True,
        help="输入的 ZARR 文件路径（例如 data_policy/StackCube_franka_200.zarr）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出的 .npy 文件路径（例如 /path/to/random_cloud.npy）",
    )

    parser.add_argument("--store_rgb", action="store_true", help="是否存储 RGB 通道")
    parser.add_argument("--store_depth", action="store_true", help="是否存储深度通道")
    parser.add_argument("--store_pnt_cloud", action="store_true", help="是否存储点云")
    parser.add_argument(
        "--seed", type=int, default=None, help="随机种子（可选），以便复现"
    )
    parser.add_argument(
        "--calculate_mean_std", action="store_true", help="是否计算深度的均值和标准差"
    )
    args = parser.parse_args()

    depth_dir = os.path.join(args.output_dir, "depth")
    pnt_cloud_dir = os.path.join(args.output_dir, "pnt_cloud")
    rgb_dir = os.path.join(args.output_dir, "rgb")

    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir, exist_ok=True)
    if not os.path.exists(pnt_cloud_dir):
        os.makedirs(pnt_cloud_dir, exist_ok=True)
    if not os.path.exists(rgb_dir):
        os.makedirs(rgb_dir, exist_ok=True)

    # 可选：设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)

    root = zarr.open_group(args.zarr_path, mode="r")
    head_cams = root["data"]["head_camera"]
    depths = root["data"]["head_camera_depth"]
    pnt_clouds = root["data"]["head_camera_pnt_cloud"]

    head_cam_exp = head_cams[0]
    depth_exp = depths[0]
    pnt_cloud_exp = pnt_clouds[0]
    if args.store_rgb:
        print(f"head_camera shape: {head_cam_exp.shape}")
        max_rgb = head_cam_exp.max().item()
        min_rgb = head_cam_exp.min().item()
        print(f"head_camera min: {min_rgb}, max: {max_rgb}")
        output_dir = os.path.join(rgb_dir, "0.png")
        iio.imwrite(output_dir, np.moveaxis(head_cam_exp, 0, -1))

    if args.store_depth:
        print(f"head_camera_depth shape: {depth_exp.shape}")
        depth_max = depth_exp.max().item()
        depth_min = depth_exp.min().item()
        print(f"head_camera_depth min: {depth_min}, max: {depth_max}")
        depth_norm = (depth_exp - depth_min) / (
            depth_max - depth_min
        )  # 先归一化到 [0,1]
        output_dir = os.path.join(depth_dir, "0.png")
        iio.imwrite(
            output_dir,
            np.moveaxis((depth_norm * 255).astype(np.uint8), 0, -1),
        )

    if args.store_pnt_cloud:
        print(f"head_camera_pnt_cloud shape: {pnt_cloud_exp.shape}")

        try:
            pcd_dataset = root["data"]["head_camera_pnt_cloud"]
        except KeyError:
            raise KeyError(
                f"在 ZARR 路径 {args.zarr_path} 中未找到 data/head_camera_pnt_cloud 数据集"
            )

        # dataset 的第一个维度是样本数量，每个样本形状为 (N,3) 或 (N,6)
        total = len(pcd_dataset)
        if total == 0:
            raise ValueError("点云数据集为空")

        for idx in tqdm(range(total)):
            your_pointcloud = pcd_dataset[idx]
            output_dir = os.path.join(pnt_cloud_dir, f"{idx}.npy")
            np.save(output_dir, your_pointcloud)

        # 可视化
        print(f"可视化第 {idx} 个点云（共 {total} 个样本）")
        visualizer.visualize_pointcloud(your_pointcloud)
    # 打开 ZARR 文件
    # 假设点云存储在 group 'data' 下的 dataset 'head_camera_pnt_cloud'
    if args.calculate_mean_std:
        depths = depths[:, :1, :, :]
        assert (
            depths.shape[1] == 1 and depths.shape[2] == 256 and depths.shape[3] == 256
        ), "depths shape should be (N, 1, H, W)"
        min_d = depths.min(axis=(1, 2, 3), keepdims=True)
        max_d = depths.max(axis=(1, 2, 3), keepdims=True)
        depths_norm = (depths - min_d) / (max_d - min_d)  # [0,1]
        print("calculate depth mean and std")
        mean_val = depths_norm.mean()
        std_val = depths_norm.std()

        print(f"Normalized depth mean: {mean_val:.6f}")
        print(f"Normalized depth std:  {std_val:.6f}")


if __name__ == "__main__":
    main()
