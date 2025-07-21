import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
import time

class JointTrajectoryActionClient(Node):
    def __init__(self):
        super().__init__('joint_trajectory_action_client')
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/fr3_arm_controller/follow_joint_trajectory'
        )


    def send_goal(self, joint_positions):
        # Wait for action server to be available
        self.action_client.wait_for_server()

        # Create goal message
        goal_msg = FollowJointTrajectory.Goal()

        # Set joint names for Franka robot (FR3)
        goal_msg.trajectory.joint_names = [
            'fr3_joint1',
            'fr3_joint2',
            'fr3_joint3',
            'fr3_joint4',
            'fr3_joint5',
            'fr3_joint6',
            'fr3_joint7'
        ]

        # Create a trajectory point with desired positions
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start.sec = 2  # Reach target position in 2 seconds

        # Add point to trajectory
        goal_msg.trajectory.points.append(point)

        # Send goal
        self.get_logger().info('Sending goal request...')
        self.action_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)

    # Create action client
    action_client = JointTrajectoryActionClient()
    joint_positions = [0.101, -0.198, -0.119, -2.573, -0.313, 2.103, 0.448]
    action_client.send_goal(joint_positions)
    time.sleep(3)  # Wait for the action to complete
    delta = [0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010]
    for i in range(10):
        # Target joint positions
        joint_positions = [x + d for x, d in zip(joint_positions, delta)]

        # Send goal
        action_client.send_goal(joint_positions)

    # Keep node alive for a moment to process goal
    time.sleep(3)

    # Clean up
    rclpy.shutdown()

if __name__ == '__main__':
    main()
