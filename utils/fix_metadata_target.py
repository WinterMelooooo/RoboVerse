import os
import json
import numpy as np


src_dir = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_libero/ExtLibero100/robot-franka"
subdirs = sorted(os.listdir(src_dir))
for subdir in subdirs:
    if subdir.startswith("demo_"):
        demo_root = os.path.join(src_dir, subdir)
        metadata_path = os.path.join(demo_root, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            robot_ee_state = np.array(metadata['robot_ee_state'])  # (T, 6)
            gripper_state = np.array(metadata['gripper_state'])  # (T, 2)
            gripper_state = abs(gripper_state)
            robot_ee_state = np.concatenate([gripper_state, robot_ee_state], axis=-1)  # (T, 8)
            action = np.array([robot_ee_state[i+1]-robot_ee_state[i] for i in range(len(robot_ee_state)-1)])
            action = np.append(action, np.zeros((1,8)), axis=0)  # (T, 8)
            action = action[1:]
            action = np.append(action, action[-1:, :], axis=0)  # (T, 8)
            metadata["joint_qpos"] = robot_ee_state.tolist()
            metadata["joint_qpos_target"] = action.tolist()
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Updated metadata in {metadata_path}")
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
