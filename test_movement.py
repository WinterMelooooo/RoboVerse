# # Copyright (c) Facebook, Inc. and its affiliates.

# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.
# import torch
# import time
# from scipy.spatial.transform import Rotation as R
# from polymetis import RobotInterface
# from polymetis import GripperInterface

# if __name__ == "__main__":
#     # Initialize robot interface
#     robot = RobotInterface(
#         ip_address="172.16.0.1",
#         port = 50051,
#     )
#     gripper = GripperInterface(
#         ip_address="172.16.0.1",
#     )
#     # Reset
#     robot.go_home()
#     # Joint impedance control
#     joint_positions = robot.get_joint_positions()
#     print("Performing joint impedance control...")
#     robot.start_joint_impedance()

#     # times = []
#     # for i in range(40):
#     #     start = time.time()
#     #     print(f"Iteration {i+1}/40")
#     #     joint_positions += torch.Tensor([0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005])
#     #     robot.update_desired_joint_positions(joint_positions)

#     #     end = time.time()
#     #     times.append(end - start)
#     #     time.sleep(4)

#     # print(f"Average time per iteration: {sum(times) / len(times):.4f} seconds")

#     gripper.grasp(grasp_width=0.08, speed=0.05, force=0.1)
#     print("Gripper grasped at width 0.08")
#     gripper.grasp(grasp_width=0.02, speed=0.05, force=0.1)
#     print("Gripper grasped at width 0.02")
#     time.sleep(3.0)
#     gripper.grasp(grasp_width=0.04, speed=0.05, force=0.1)
#     print("Gripper grasped at width 0.04")

#     # gripper.grasp(grasp_width=0.08, speed=0.05, force=0.1)
#     # gripper.goto(width=0.08, speed=0.05, force=0.1)
#     # gripper.grasp(speed=0.05, force=0.1)
#     # gripper_state = gripper.get_state()
#     # time.sleep(2.0)
#     # gripper.goto(width=0.01, speed=0.05, force=0.1)
#     # gripper.grasp(speed=0.05, force=0.1)

#     # robot.terminate_current_policy()


import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
from franka_msgs.action import Homing, Move, Grasp
from franka_msgs.msg import GraspEpsilon
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.executors import SingleThreadedExecutor
import time
from roboverse_learn.algorithms.utils.franka_ros import FrankaRobot
def main(args=None):
    rclpy.init(args=args)
    robot_client = FrankaRobot()
    robot_client.do_homing()
    delta = [-0.000, -0.000,  0.100, 0.1010, 0.1010, 0.1010, 0.1010, 0.1010, 0.1010]
    for i in range(10):
        # Target joint positions
        joint_positions = [x + d for x, d in zip(robot_client.get_state(), delta)]

        # Send goal
        robot_client.goto(joint_positions)

    # Clean up
    rclpy.shutdown()


if __name__ == '__main__':
    main()
