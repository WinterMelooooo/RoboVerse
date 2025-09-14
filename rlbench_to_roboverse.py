
import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import json
import argparse
from tqdm import tqdm


DEFAULT_RLBENCH_DIR = "/home/ghr/yktang/RoboVerse/third_party/rlbench-18-tasks/data/train"
DEFAULT_ROBOVERSE_DIR = "/home/ghr/yktang/RoboVerse/roboverse_demo"

TASK_NAME = "sweep_to_dustpan_of_size"

def convert_depth_batch(depths):

    depth_raw = (
        depths[..., 2]
        + depths[..., 1] * 256
        + depths[..., 0] * 256**2
    )  # shape [N, H, W]

    depth_normalized = depth_raw / (256**3 - 1)  # shape [N, H, W]
    min_per_frame = depth_normalized.min(axis=(1, 2), keepdims=True)
    max_per_frame = depth_normalized.max(axis=(1, 2), keepdims=True)
    depth_per_frame_norm = (depth_normalized - min_per_frame) / (
        max_per_frame - min_per_frame
    )
    return depth_per_frame_norm

def plot_depth_npy(depth_npy):

    plt.figure()
    plt.imshow(depth_npy, cmap='viridis')
    plt.axis('off')
    plt.title('Depth NPY')
    plt.show()

def convert_single_demo(demo_dir, dest_dir):
    rgbs = []
    depths = []
    joint_qposes = []
    depth_min = []
    depth_max = []
    cam_intr = []
    cam_extr = []
    robot_root_states = []
    rgb_dir = os.path.join(demo_dir, "front_rgb")
    depth_dir = os.path.join(demo_dir, "front_depth")
    low_dim_obs_path = os.path.join(demo_dir, "low_dim_obs.pkl")
    rgbs_names = sorted(os.listdir(rgb_dir))
    depths_names = sorted(os.listdir(depth_dir))
    total_frames = len(rgbs_names)

    with open(low_dim_obs_path, "rb") as f:
        low_dim_data = pickle.load(f)

    for frame_idx in range(total_frames):
        rgb_path = os.path.join(rgb_dir, rgbs_names[frame_idx])
        depth_path = os.path.join(depth_dir, depths_names[frame_idx])
        rgb = iio.imread(rgb_path)
        depth = iio.imread(depth_path)
        rgbs.append(rgb)
        depths.append(depth)

        low_dim = low_dim_data[frame_idx]
        joint_qpos = low_dim.joint_positions
        gripper_qpos = low_dim.gripper_joint_positions
        joint_qpos_full = gripper_qpos.tolist() + joint_qpos.tolist()
        joint_qposes.append(joint_qpos_full)
        intr = low_dim.misc['front_camera_intrinsics']
        extr = low_dim.misc['front_camera_extrinsics']
        intr[0,0] = -intr[0,0]
        intr[1,1] = -intr[1,1]
        extr = np.linalg.inv(extr)

        cam_intr.append(intr.tolist())
        cam_extr.append(extr.tolist())
        depth_min.append(low_dim.misc['front_camera_near'])
        depth_max.append(low_dim.misc['front_camera_far'])

        robot_root_state = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        robot_root_states.append(robot_root_state)
    rgbs = np.array(rgbs)
    depths = np.array(depths)
    depths = convert_depth_batch(depths)
    joint_qpos_target = joint_qposes[1:] + [joint_qposes[-1]]
    save_demo(
        rgbs=rgbs,
        depths=depths,
        joint_qpos=joint_qposes,
        joint_qpos_target=joint_qpos_target,
        cam_intr=cam_intr,
        cam_extr=cam_extr,
        depth_min=depth_min,
        depth_max=depth_max,
        robot_root_states=robot_root_states,
        dest_dir=dest_dir,
    )
    print(f"Converted and saved demo to {dest_dir}, total frames: {total_frames}")


def save_demo(rgbs, depths, joint_qpos, joint_qpos_target, cam_intr, cam_extr, depth_min, depth_max, robot_root_states, dest_dir):
    iio.mimsave(os.path.join(dest_dir, "rgb.mp4"), rgbs, fps=30, quality=10)
    iio.mimsave(
        os.path.join(dest_dir, "depth_uint8.mp4"),
        [(depth * 255).astype(np.uint8) for depth in depths],
        fps=30,
        quality=10,
    )
    jsondata = {
        "depth_min": depth_min,
        "depth_max": depth_max,
        "cam_intr": cam_intr,
        "cam_extr": cam_extr,
        "joint_qpos": joint_qpos,
        "robot_root_state": robot_root_states,
        "joint_qpos_target": [],
    }

    for frame_idx in range(len(joint_qpos)):
        if frame_idx < len(joint_qpos) - 1:
            next_joint_qpos_target = joint_qpos_target[frame_idx + 1]
            jsondata["joint_qpos_target"].append(next_joint_qpos_target)
        else:
            jsondata["joint_qpos_target"].append(joint_qpos_target[frame_idx])
    json.dump(jsondata, open(os.path.join(dest_dir, "metadata.json"), "w"))

def rename_dataset(dataset_dir):
    episode_dirs = sorted([d for d in os.listdir(dataset_dir) if d.startswith("episode")])
    for idx, episode_dir in enumerate(episode_dirs):
        print(f"Renaming episode {episode_dir}")
        src = os.path.join(dataset_dir, episode_dir)
        subdirs = os.listdir(src)
        for subdir in subdirs:
            if not os.path.isdir(os.path.join(src, subdir)):
                continue
            print(f"\tProcessing subdir {subdir}")
            dir = os.path.join(src, subdir)
            files = os.listdir(dir)
            for file in files:
                if file.endswith(".png"):
                    old_path = os.path.join(dir, file)
                    frame_idx = int(file.split(".")[0])
                    new_filename = f"{frame_idx:04d}.png"
                    new_path = os.path.join(dir, new_filename)
                    os.rename(old_path, new_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_idx", type=int)
    args = parser.parse_args()
    demo_idx = args.demo_idx

    rlbench_task_dir = os.path.join(DEFAULT_RLBENCH_DIR, TASK_NAME, "all_variations", "episodes")
    roboverse_task_dir = os.path.join(DEFAULT_ROBOVERSE_DIR, "demo_rlbench", TASK_NAME, "robot_franka")
    os.makedirs(roboverse_task_dir, exist_ok=True)
    demo_dir = os.path.join(rlbench_task_dir, f"episode{demo_idx}")
    dest_dir = os.path.join(roboverse_task_dir, f"demo_{demo_idx:04d}")
    os.makedirs(dest_dir, exist_ok=True)
    #rename_dataset(rlbench_task_dir)
    convert_single_demo(demo_dir, dest_dir)


if __name__ == "__main__":
    main()
