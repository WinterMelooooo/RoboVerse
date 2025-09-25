import os
import json

dir = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_realworld/RealworldLiberoPickButter_fixed_butter/robot-franka"
target_finger_pos = 0.03975328803062439
folders = os.listdir(dir)
folders = sorted(folders)
for folder in folders:
    folder_path = os.path.join(dir, folder)
    metadata_path = os.path.join(folder_path, "metadata.json")
    backup_path = os.path.join(folder_path, "metadata.json.bak")
    if os.path.exists(backup_path):
        print(f"Found backup for {folder_path}")
        os.remove(metadata_path)
        os.rename(backup_path, metadata_path)
        #continue
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"{metadata_path} not found")
    metadata = json.load(open(metadata_path, "r"))
    joint_states = metadata["joint_qpos"]
    if joint_states[0][0] < 0.03:
        print(f"Fixing {folder_path}")
        delta = target_finger_pos - joint_states[0][0]
        # Backup
        os.rename(metadata_path, metadata_path + ".bak")
        starting_finger_pos = joint_states[0][0]
        for i in range(len(joint_states)):
            #current_finger_pos = joint_states[i][0]
            # if abs(current_finger_pos - starting_finger_pos) < 0.01:
            #     joint_states[i][0] = target_finger_pos
            #     joint_states[i][1] = target_finger_pos
            # else:
            #     break
            joint_states[i][0] += delta
            joint_states[i][1] += delta
        metadata["joint_qpos"] = joint_states
        json.dump(metadata, open(metadata_path, "w"), indent=4)
        print(f"Fixed {folder_path}")
    else:
        print(f"{folder_path} is fine")
