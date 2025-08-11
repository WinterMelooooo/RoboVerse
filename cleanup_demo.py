import json
import os

from tqdm import tqdm

data_input_seq = [
    "panda_finger_joint1",
    "panda_finger_joint2",
    "panda_joint1",
    "panda_joint3",
    "panda_joint6",
    "panda_joint7",
    "panda_joint2",
    "panda_joint4",
    "panda_joint5",
]

desire_data_seq = [
    "panda_finger_joint1",
    "panda_finger_joint2",
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

demo_root = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_realworld/RealworldLiberoPickButter/robot-franka"
subdirs = os.listdir(demo_root)
subdirs.sort()

for i in tqdm(range(len(subdirs))):
    subdir = subdirs[i]
    metadata_path = os.path.join(demo_root, subdir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            data = json.load(f)
    else:
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    target_actions = data["joint_qpos_target"]
    actions = data["joint_qpos"]
    actions = [ep[0] for ep in actions]
    data["joint_qpos"] = actions
    # new_actions = []
    # new_target_actions = []
    # for idx, action in enumerate(actions):
    #     action = actions[idx]
    #     target_action = target_actions[idx]
    #     action_dict = {k: v for k, v in zip(data_input_seq, action)}
    #     target_action_dict = {k: v for k, v in zip(data_input_seq, target_action)}
    #     action_list = [action_dict[k] for k in desire_data_seq]
    #     target_action_list = [target_action_dict[k] for k in desire_data_seq]
    #     new_actions.append(action_list)
    #     new_target_actions.append(target_action_list)

    # data["joint_qpos"] = new_actions
    # data["joint_qpos_target"] = new_target_actions

    with open(metadata_path, "w") as f:
        json.dump(data, f, indent=4)
