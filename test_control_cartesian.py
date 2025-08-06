#!/usr/bin/env python3
# control_cartesian.py
# 一个 ROS2 节点，用于向 Franka 笛卡尔位姿示例控制器发送命令并获取当前末端执行器位姿

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped

class CartesianCommander(Node):
    def __init__(self):
        super().__init__('cartesian_commander')
        # 发布器：向控制器的 command 接口话题发布 Float64MultiArray
        self.pub = self.create_publisher(
            Float64MultiArray,
            '/cartesian_pose_example_controller/command',
            10
        )

        # TF2 缓冲区和监听器，用于获取当前末端执行器位姿
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 声明位姿参数（默认值）
        self.declare_parameter('x', 0.35)
        self.declare_parameter('y', 0.35)
        self.declare_parameter('z', 0.35)

        # 读取参数并构建 4x4 齐次变换矩阵
        x = self.get_parameter('x').value
        y = self.get_parameter('y').value
        z = self.get_parameter('z').value
        pose = np.eye(4)
        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z

        # 将矩阵按行优先展开为列表
        self.command_data = pose.flatten().tolist()

        # 定时器：每秒执行一次发送命令和获取当前位姿
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info(f"初始化完成：目标位置=({x:.3f}, {y:.3f}, {z:.3f})")

    def send_command(self):
        msg = Float64MultiArray()
        msg.data = self.command_data
        self.pub.publish(msg)
        self.get_logger().info('已发布笛卡尔位姿命令')

    def get_current_pose(self):
        try:
            # 查询 base 到末端执行器链接 fr3_link8 的变换，可根据 URDF 修改
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                'base',           # 源坐标系
                'fr3_link8',      # 目标坐标系（末端执行器）
                rclpy.time.Time() # 获取最新时间戳的变换
            )
            pos = trans.transform.translation
            ori = trans.transform.rotation
            self.get_logger().info(
                f"当前末端位姿 - 位置: (x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}), "
                f"姿态: (x={ori.x:.3f}, y={ori.y:.3f}, z={ori.z:.3f}, w={ori.w:.3f})"
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"TF 查找失败: {e}")

    def timer_callback(self):
        # 先发送命令，再获取并打印当前末端执行器位姿
        self.send_command()
        self.get_current_pose()


def main(args=None):
    rclpy.init(args=args)
    node = CartesianCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
