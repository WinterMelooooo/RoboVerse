#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient

from franka_msgs.action import Homing, Move, Grasp
from franka_msgs.msg import GraspEpsilon
from franka_msgs.srv import SetForceTorqueCollisionBehavior


class GripperActionClient(Node):
    def __init__(self):
        super().__init__('gripper_action_client')
        cb_group = ReentrantCallbackGroup()

        # Action Clients
        self.homing_client = ActionClient(self, Homing,  '/gripper/franka_gripper/homing', callback_group=cb_group)
        self.move_client   = ActionClient(self, Move,    '/gripper/franka_gripper/move',   callback_group=cb_group)
        self.grasp_client  = ActionClient(self, Grasp,   '/gripper/franka_gripper/grasp',  callback_group=cb_group)
        #self.test_client = ActionClient(self, )
    def do_homing(self) -> bool:
        """执行 Homing 校准"""
        self.get_logger().info('>>> 等待 Homing Server...')
        if not self.homing_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Homing Server 不可用！')
            return False

        goal = Homing.Goal()
        self.get_logger().info('>>> 发送 Homing 请求...')
        fut = self.homing_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if not gh.accepted:
            self.get_logger().error('Homing 请求被拒绝！')
            return False

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        self.get_logger().info('>>> Homing 完成')
        return True

    def do_move(self, width: float, speed: float = 0.05) -> bool:
        """使用 Move Action，开启/闭合到指定宽度"""
        # 先调高碰撞阈值
        self.set_collision_behavior()

        self.get_logger().info(f'>>> 等待 Move Server... 目标宽度={width:.3f} m')
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Move Server 不可用！')
            return False

        goal = Move.Goal()
        goal.width = width
        goal.speed = speed

        self.get_logger().info('>>> 发送 Move 请求...')
        fut = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if not gh.accepted:
            self.get_logger().error('Move 请求被拒绝！')
            return False

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        self.get_logger().info(f'>>> Move 完成，返回：{res_fut.result().result}')
        return True

    def do_grasp(self, width: float, speed: float = 0.05, force: float = 20.0) -> bool:
        """使用 Grasp Action，带容差 epsilon 抓取并保持力"""

        self.get_logger().info(f'>>> 等待 Grasp Server... 目标宽度={width:.3f} m')
        if not self.grasp_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Grasp Server 不可用！')
            return False

        goal = Grasp.Goal()
        goal.width = width
        goal.epsilon = GraspEpsilon(inner=width, outer=width)
        goal.speed = speed
        goal.force = force

        def feedback_cb(msg):
            cur = msg.feedback.current_width
            self.get_logger().info(f'    [反馈] 当前宽度 = {cur:.4f} m')

        self.get_logger().info('>>> 发送 Grasp 请求...')
        fut = self.grasp_client.send_goal_async(goal, feedback_callback=feedback_cb)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if not gh.accepted:
            self.get_logger().error('Grasp 请求被拒绝！')
            return False

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        result = res_fut.result().result
        self.get_logger().info(f'>>> Grasp 完成：success={result.success}, error="{result.error}"')
        return result.success


def main(args=None):
    rclpy.init(args=args)
    client = GripperActionClient()

    client.do_homing()

    client.do_grasp(0.00, speed=0.1)
    client.do_grasp(0.01, speed=0.1)
    client.do_grasp(0.00, speed=0.1)



if __name__ == '__main__':
    main()
