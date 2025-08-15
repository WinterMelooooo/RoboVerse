import json
import os
import time
import imageio as iio
import torch
import rclpy
from roboverse_learn.algorithms.utils.franka_ros import FrankaRobot

rclpy.init()
robot = FrankaRobot()
metadata_path = "/home/user/yktang/RoboVerse/roboverse_demo/demo-realworld/RealworldLiberoPickButter/robot-franka/demo_0000/metadata.json"

data = json.load(open(metadata_path, "r"))
actions = data["joint_qpos"]
# data_input_seq = [
#     "panda_finger_joint1",
#     "panda_finger_joint2",
#     "panda_joint1",
#     "panda_joint3",
#     "panda_joint6",
#     "panda_joint7",
#     "panda_joint2",
#     "panda_joint4",
#     "panda_joint5",
# ]
data_input_seq = [
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

command_per_sec = 10

for action in actions:
    start = time.time()
    robot.goto(action)
    end = time.time()
    time.sleep(max(0, 1.0 / command_per_sec - (end - start)))
