#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class FrankaStateReader(Node):
    def __init__(self):
        super().__init__('franka_state_reader')
        self.create_subscription(
            JointState,
            '/joint_states',
            self.callback_joint_states,
            10
        )
        self.joint_pos = None
        self.gripper_width = None

    def callback_joint_states(self, msg: JointState):
        names = msg.name
        pos   = msg.position
        # 1) 机械臂 7 关节
        arm_joints = []
        for jn in [
            'fr3_joint1','fr3_joint2','fr3_joint3',
            'fr3_joint4','fr3_joint5','fr3_joint6','fr3_joint7'
        ]:
            arm_joints.append(pos[names.index(jn)] if jn in names else None)

        # 2) 爪子宽度
        width = None
        for jn in ['fr3_finger_joint1', 'fr3_finger_joint2']:
            if jn in names:
                width = 2*pos[names.index(jn)]
                break

        self.joint_pos = arm_joints
        self.gripper_width = width
        return

    def get_state(self):
        return self.joint_pos, self.gripper_width

def main():
    rclpy.init()
    node = FrankaStateReader()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
