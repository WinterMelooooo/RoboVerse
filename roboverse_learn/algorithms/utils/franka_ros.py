try:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from franka_msgs.action import Homing, Move, Grasp
    from franka_msgs.msg import GraspEpsilon
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectoryPoint
    from rclpy.executors import SingleThreadedExecutor
    import zmq
    import time
except:
    pass

class FrankaArm(Node):
    def __init__(self):
        super().__init__('joint_trajectory_action_client')
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/fr3_arm_controller/follow_joint_trajectory'
        )
        # self.home = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]
        # self.home = [0.0, 0.42, -0.013, -1.975, -0.013, 2.366, 0.79] # PickCube
        self.home = [0.0, 0.175, 0.0, -0.873, 0.0, 1.221, 0.785]  # CloseBox

    def goto(self, joint_positions):
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

    def get_state(self):
        """
        返回当前 7 关节的角度 [rad]
        如果还没收到 joint_states，返回 None
        """
        return self._latest_js

    def do_homing(self):
        self.goto(self.home)

class FrankaGripper(Node):
    def __init__(self):
        super().__init__('gripper_action_client')

        # Action Clients
        self.homing_client = ActionClient(self, Homing,  '/gripper/franka_gripper/homing')
        self.move_client   = ActionClient(self, Move,    '/gripper/franka_gripper/move')
        self.grasp_client  = ActionClient(self, Grasp,   '/gripper/franka_gripper/grasp')
        self.threshold = 0.001
    def do_homing(self) -> bool:
        time.sleep(1.0)  # 等待一会儿，确保 Action Server 已经启动
        self.do_move(0.08)
        time
        # """执行 Homing 校准"""
        # self.get_logger().info('>>> 等待 Homing Server...')
        # if not self.homing_client.wait_for_server(timeout_sec=5.0):
        #     self.get_logger().error('Homing Server 不可用！')
        #     return False

        # goal = Homing.Goal()
        # self.get_logger().info(f'>>> 发送 Homing 请求...')
        # fut = self.homing_client.send_goal_async(goal)
        # rclpy.spin_until_future_complete(self, fut)
        # gh = fut.result()
        # if not gh.accepted:
        #     self.get_logger().error('Homing 请求被拒绝！')
        #     return False

        # res_fut = gh.get_result_async()
        # rclpy.spin_until_future_complete(self, res_fut)
        # self.get_logger().info('>>> Homing 完成')
        return True

    def do_move(self, width: float, speed: float = 0.10) -> bool:
        """使用 Move Action，开启/闭合到指定宽度"""
        # 先调高碰撞阈值

        self.get_logger().info(f'>>> 等待 Move Server... 目标宽度={width:.3f} m')
        if not self.move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Move Server 不可用！')
            return False

        goal = Move.Goal()
        goal.width = width
        goal.speed = speed

        self.get_logger().info('>>> 发送 Move 请求...')
        fut = self.move_client.send_goal_async(goal)
        # rclpy.spin_until_future_complete(self, fut)
        # gh = fut.result()
        # if not gh.accepted:
        #     self.get_logger().error('Move 请求被拒绝！')
        #     return False

        # res_fut = gh.get_result_async()
        # rclpy.spin_until_future_complete(self, res_fut)
        # self.get_logger().info(f'>>> Move 完成，返回：{res_fut.result().result}')
        # return True

    def do_grasp(self, width: float, speed: float = 1.0, force: float = 20.0) -> bool:
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
        # rclpy.spin_until_future_complete(self, fut)
        # gh = fut.result()
        # if not gh.accepted:
        #     self.get_logger().error('Grasp 请求被拒绝！')
        #     return False

        # res_fut = gh.get_result_async()
        # rclpy.spin_until_future_complete(self, res_fut)
        # result = res_fut.result().result
        # self.get_logger().info(f'>>> Grasp 完成：success={result.success}, error="{result.error}"')
        # return result.success

    def goto(self, width: float, current_width: float) -> bool:
        if width < current_width - self.threshold:
            self.do_grasp(width)
        else:
            self.do_move(width)

class FrankaRobot():
    def __init__(self):
        self.arm_client = FrankaArm()
        self.gripper_client = FrankaGripper()
        self._latest_js = None
        self.arm_client.create_subscription(
            JointState,
            '/joint_states',        # FR3 的 controller_manager 通常在这里发布
            self._js_callback,
            10
        )
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.arm_client)
        self.executor.add_node(self.gripper_client)
        self.desired_seq = [
            "fr3_finger_joint1",
            "fr3_finger_joint2",
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7"
        ]

    def _js_callback(self, msg: JointState):
        state_dict = dict(zip(msg.name, msg.position))
        state = [state_dict[joint] for joint in self.desired_seq]
        self._latest_js = state

    def get_state(self):
        self.executor.spin_once(timeout_sec=0.5)
        state = self._latest_js
        return state

    def goto(self, goal):
        if not len(goal) == len(self.desired_seq):
            raise ValueError(f"目标关节数 {len(goal)} 与期望 {len(self.desired_seq)} 不匹配")
        current_gripper_state = self.get_state()[0:2]
        current_gripper_width = sum(current_gripper_state)
        joint_pos = goal[-7:]
        gripper_width = sum(goal[0:2])
        self.arm_client.goto(joint_pos)
        self.gripper_client.goto(gripper_width, current_gripper_width)

    def do_homing(self):
        self.arm_client.do_homing()
        self.gripper_client.do_homing()
        # while self.get_state() is None:
        #     self.executor.spin_once(timeout_sec=0.1)
        #     time.sleep(0.01)
        print("已收到第一条关节状态：", self.get_state())

class FrankaRobotServer():
    def __init__(self, socket_number = 5555):
        self.robot = FrankaRobot()
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://127.0.0.1:{socket_number}")
        print(f"Franka Robot Server Listening on tcp://127.0.0.1:{socket_number}")
        self.socket = socket

    def run(self):
        while True:
            message = self.socket.recv_json()
            print(f"Received request: {message}") # {"command": xxx, .....}
            if message['command'] == 'goto':
                self.robot.goto(message['goal'])
                response = {'status': 'success', 'message': 'Goal reached'}
            elif message['command'] == 'homing':
                self.robot.do_homing()
                response = {'status': 'success', 'message': 'Homing completed'}
            elif message['command'] == 'get_state':
                state = self.robot.get_state()
                response = {'status': 'success', 'state': state}
            else:
                response = {'status': 'error', 'message': 'Unknown action'}
            self.socket.send_json(response)
