#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

class FrankaStateReader(Node):
    def __init__(self):
        super().__init__('franka_state_reader')
        self.joint_pos = None
        self.gripper_width = None
        self.target_joints = None
        self.target_gripper_width = None
        self._joint_names = [
            'fr3_joint1','fr3_joint2','fr3_joint3',
            'fr3_joint4','fr3_joint5','fr3_joint6','fr3_joint7'
        ]
        self.create_subscription(
            JointState,
            '/joint_states',
            self.callback_joint_states,
            10
        )
        # 订阅目标关节角
        self.create_subscription(
            JointState, 'gello/joint_states', self.callback_target_joint_states, 10
        )
        # 订阅目标夹爪百分比（0~1）
        self.create_subscription(
            Float32, 'gripper/gripper_client/target_gripper_width_percent', self.callback_target_gripper, 10
        )

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

    def callback_target_joint_states(self, msg: JointState):
        name2idx = {n:i for i,n in enumerate(msg.name)}
        try:
            self.target_joints = [float(msg.position[name2idx[jn]]) for jn in self._joint_names]
        except Exception as e:
            raise ValueError(f"Target joint names {self._joint_names} not found in message names {msg.name}\nGot error: {e}")

    def callback_target_gripper(self, msg: Float32):
        if msg.data < 0 or msg.data > 1:
            raise ValueError("Gripper width must be between 0 and 1")
        target_gripper_percentage = msg.data
        self.target_gripper_width = target_gripper_percentage * 0.08

    def get_state(self):
        return self.joint_pos, self.gripper_width

    def get_target_state(self):
        return self.target_joints, self.target_gripper_width

def main():
    rclpy.init()
    node = FrankaStateReader()
    try:
        import time
        last_print = 0.0
        while rclpy.ok():
            # 处理一次回调（订阅到的消息会更新 node 的属性）
            rclpy.spin_once(node, timeout_sec=0.1)

            # 节流打印
            if time.time() - last_print > 0.5:
                joints, grip_w = node.get_state()
                tgt_joints, tgt_grip_w = node.get_target_state()

                if joints is not None and grip_w is not None:
                    print("[STATE ] joints:", [f"{x:.3f}" for x in joints],
                            "gripper_w:", f"{grip_w:.3f} m")
                if tgt_joints is not None and tgt_grip_w is not None:
                    print("[TARGET] joints:", [f"{x:.3f}" for x in tgt_joints],
                            "gripper_w:", f"{tgt_grip_w:.3f} m")

                last_print = time.time()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
