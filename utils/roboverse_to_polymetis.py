import os
import torch
import imageio as iio
import json
import argparse


def convert_single_demo(demo_dir, dest_dir):
    rgb_path = os.path.join(demo_dir, "rgb.mp4")
    metadata_path = os.path.join(demo_dir, "metadata.json")
    rgbs = iio.get_reader(rgb_path)
    metadata = json.load(open(metadata_path, "r"))
    for frame_idx, rgb in enumerate(rgbs):
        rgb_name = f"color_image_{frame_idx:04d}.png"
        iio.imwrite(os.path.join(dest_dir, rgb_name), rgb)
        joint_qpos = metadata["joint_qpos"][frame_idx]
        joint_qpos_target = metadata["joint_qpos_target"][frame_idx]
        state = {
            "joint_pos": torch.tensor(joint_qpos[2:]),
            "joint_pos_target": torch.tensor(joint_qpos_target[2:]),
            "width": joint_qpos[0] + joint_qpos[1],
            "width_target": joint_qpos_target[0] + joint_qpos_target[1],
        }
        state_name = f"state_{frame_idx:04d}.pt"
        torch.save(state, os.path.join(dest_dir, state_name))
    print(f"Converted {demo_dir} to {dest_dir}")
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", type=str, required=True, help="Path to the source directory containing RoboVerse demos.")
    parser.add_argument("--dest_dir", type=str, required=True, help="Path to the destination directory for Polymetis formatted demos.")
    args = parser.parse_args()

    demo_dir = args.demo_dir
    dest_dir = args.dest_dir

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    convert_single_demo(demo_dir, dest_dir)

if __name__ == "__main__":
    main()
