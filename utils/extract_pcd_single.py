import os
import json
import cv2
import numpy as np
import argparse
import sys
sys.path.append("/home/ghr/yktang/RoboVerse")
from roboverse_learn.algorithms.utils.pnt_cloud_getter import PntCloudGetter
pnt_cloud_getter = PntCloudGetter("ExtLibero100", use_point_crop=True)

def read_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频：{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("视频帧数未知或为 0")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    frames = np.array(frames)
    frames = frames[:, :, :, ::-1]  # BGR to RGB
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_root_dir', type=str, required=True, help='Path to the demo root directory')
    args = parser.parse_args()

    demo_root_path = args.demo_root_dir
    rgb_video_path = os.path.join(demo_root_path, "rgb.mp4")
    depth_video_path = os.path.join(demo_root_path, "depth_uint8.mp4")
    metadata_path = os.path.join(demo_root_path, "metadata.json")

    data = json.load(open(metadata_path, "r"))
    rgbs = read_frames(rgb_video_path)
    depths = read_frames(depth_video_path)
    total_frames = rgbs.shape[0]
    depths = depths[..., 0] / 255.0  # (256,256) [0,1]
    if (not depths.min() < 0.2) or (not depths.max() > 0.8):
        raise ValueError(
            f"Depth values are not in the expected range [0, 1]. min: {depths.min()}, max: {depths.max()}"
        )
    point_clouds = []
    for idx in range(total_frames):
        cam_intr = np.array(data["cam_intr"][idx])
        cam_extr = np.array(data["cam_extr"][idx])
        if not cam_intr.size or not cam_extr.size:
            print(f"Cam intr and extr are empty for index {idx}. Using default values.")
        rgb = rgbs[idx]
        depth = depths[idx]
        depth_min = data["depth_min"][idx]
        depth_max = data["depth_max"][idx]
        # depth_meter = depth_min / (1 - depth * (1 - depth_min / depth_max)) # Use this for mujoco
        depth_meter = depth_min + (depth.astype(np.float32)) * (depth_max - depth_min)
        #depth_meter = clean_depth_edges(depth_meter, tau_abs=0.005, tau_rel=0.05, dilate_px=1, invalid_val=0.0)
        pnt_cloud = pnt_cloud_getter.get_point_cloud(
            rgb,
            np.ascontiguousarray(depth_meter).astype(np.float32),
            cam_intr,
            cam_extr,
        )
        point_clouds.append(pnt_cloud)

    point_clouds = np.array(point_clouds)  # (T, N, 8)
    np.save(os.path.join(demo_root_path, "pointclouds.npy"), point_clouds)
    print(f"Saved point clouds to {os.path.join(demo_root_path, 'pointclouds.npy')}")

if __name__ == '__main__':
    main()
