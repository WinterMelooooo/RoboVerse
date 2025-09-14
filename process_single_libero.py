import os
import json
import numpy as np
import argparse


exp_demo_dir = "/scratch/current/ghr/szang/szang-workspace/SyntheticVLA-mini/experiments/robot/libero/datasets/libero_90_no_noops_dp/KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet/demo_0000"
def process_single_demo(demo_dir):
    metadata = os.path.join(demo_dir, "metadata.json")
    with open(metadata, "r") as f:
        data = json.load(f)
    cam_extr = data["cam_extr"]
    cam_extr = np.array(cam_extr)
    cam_extr[:, 0, 3] += 0.66
    cam_extr[:, 2, 3] -= 0.9
    cam_extr = np.linalg.inv(cam_extr)
    data["cam_extr"] = cam_extr.tolist()
    gripper_state = data["gripper_state"]
    gripper_state = np.array(gripper_state)
    joint_qpos = data["joint_qpos"]
    joint_qpos = np.array(joint_qpos)
    gripper_state = abs(gripper_state)
    joint_state = np.concatenate([gripper_state, joint_qpos], axis=1)
    joint_state_target = joint_state[1:]
    joint_state_target = np.concatenate([joint_state_target, joint_state[-1:]], axis=0)
    data["joint_qpos"] = joint_state.tolist()
    data["joint_qpos_target"] = joint_state_target.tolist()
    with open(metadata, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Processed {demo_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_dir', type=str, default=exp_demo_dir)
    args = parser.parse_args()
    process_single_demo(args.demo_dir)


if __name__ == '__main__':
    main()
