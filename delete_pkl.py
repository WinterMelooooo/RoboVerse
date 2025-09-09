import os
import json

dir = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_realworld/RealworldLiberoPickButter_fixed_butter/robot-franka"
target_finger_pos = 0.03975328803062439
folders = os.listdir(dir)
folders = sorted(folders)
for folder in folders:
    folder_path = os.path.join(dir, folder)
    metadata_path = os.path.join(folder_path, "metadata.pkl")
    os.remove(metadata_path)
