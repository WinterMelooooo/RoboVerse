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
    from std_msgs.msg import Float64MultiArray
    from rclpy.duration import Duration
    import tf2_ros
    from geometry_msgs.msg import TransformStamped

except:
    pass
from pytorch_kinematics import Transform3d, matrix_to_quaternion
import pytorch_kinematics as pk
import os
import torch
import zmq
import time
import xml.etree.ElementTree as ET
import numpy as np
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
        # self.home = [0.0, 0.175, 0.0, -0.873, 0.0, 1.221, 0.785]  # CloseBox
        # self.home_degrees = [0, 0, 0, -90, 0, 90, -135] # PickButter
        self.home = [-0.057557623295429294, 0.00018949155714934572, -0.010052990086534465, -1.5410414293429622, -0.03143259380432439, 1.5459539128296862, -2.462186588172044]
        self.init_ee_solver()

    def init_ee_solver(self):
        urdf_path = "/ros2_ws/fr3.urdf"
        urdf_xml = open(urdf_path, 'r').read()

        self.chain = pk.build_serial_chain_from_urdf(urdf_xml, "fr3_link7")

        lim = torch.tensor(self.chain.get_joint_limits(), device=torch.device('cpu'))  # (DOF,2)
        self.ik = pk.PseudoInverseIK(
            self.chain,
            max_iterations=50,
            num_retries=10,
            joint_limits=lim.T,
            early_stopping_any_converged=True,
            early_stopping_no_improvement="all",
            debug=False,
            lr=0.2
        )


    def _select_solution(self, sol):
        solutions = sol.solutions[0]  # (num_retries, DOF)
        converged = sol.converged[0]  # (num_retries,)
        for idx, ok in enumerate(converged.tolist()):
            if ok:
                return solutions[idx]
        errs = sol.err_pos[0] + sol.err_rot[0]
        best = torch.argmin(errs)
        return solutions[best]

    def goto_joint_degree(self, joint_degrees):
        if len(joint_degrees) != 7:
            raise ValueError(f"Expected 7 joint angles in degrees, got {len(joint_degrees)}")
        joint_radians = [angle * (3.141592653589793 / 180.0) for angle in joint_degrees]
        self.get_logger().info(f"Converting degrees to radians: {joint_radians}")
        self.goto(joint_radians)

    def goto(self, joint_positions, time_to_reach=0.5):
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
        point.time_from_start = Duration(seconds=float(time_to_reach)).to_msg()

        # Add point to trajectory
        goal_msg.trajectory.points.append(point)

        # Send goal
        self.get_logger().info('Sending goal request...')
        self.action_client.send_goal_async(goal_msg)

    def get_state(self):
        return self._latest_js

    def do_homing(self):
        if hasattr(self, 'home_degrees'):
            self.goto_joint_degree(self.home_degrees)
        elif hasattr(self, 'home'):
            self.goto(self.home, time_to_reach=5.0)

    def _rotmat_to_quat(self, R: np.ndarray) -> np.ndarray:
        m = R
        # 计算
        trace = m[0,0] + m[1,1] + m[2,2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (m[2,1] - m[1,2]) * s
            qy = (m[0,2] - m[2,0]) * s
            qz = (m[1,0] - m[0,1]) * s
        else:
            if m[0,0] > m[1,1] and m[0,0] > m[2,2]:
                s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
                qw = (m[2,1] - m[1,2]) / s
                qx = 0.25 * s
                qy = (m[0,1] + m[1,0]) / s
                qz = (m[0,2] + m[2,0]) / s
            elif m[1,1] > m[2,2]:
                s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
                qw = (m[0,2] - m[2,0]) / s
                qx = (m[0,1] + m[1,0]) / s
                qy = 0.25 * s
                qz = (m[1,2] + m[2,1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
                qw = (m[1,0] - m[0,1]) / s
                qx = (m[0,2] + m[2,0]) / s
                qy = (m[1,2] + m[2,1]) / s
                qz = 0.25 * s
        quat = np.array([qx, qy, qz, qw], dtype=np.float32)
        return quat


    def goto_ee_pos(self, pos16, duration=5.0, rate_hz=10):
        arr = torch.tensor(pos16, dtype=torch.float32).reshape(4, 4)
        position = arr[:3, 3]      # (3,)
        R = arr[:3, :3]            # (3,3)
        goal_tf = Transform3d(
            pos=position,
            rot=R,
            device=position.device
        )
        print('Solving IK')
        sol = self.ik.solve(goal_tf)
        joint_pos = self._select_solution(sol)  # (7,) tensor
        print(f"IK Result: {joint_pos}")
        self.goto(joint_pos.tolist(), time_to_reach=10)


class FrankaGripper(Node):
    def __init__(self):
        super().__init__('gripper_action_client')

        # Action Clients
        self.homing_client = ActionClient(self, Homing,  '/franka_gripper/homing')
        self.move_client   = ActionClient(self, Move,    '/franka_gripper/move')
        self.grasp_client  = ActionClient(self, Grasp,   '/franka_gripper/grasp')
        self.threshold = 0.001
    def do_homing(self) -> bool:
        time.sleep(1.0)
        self.do_move(0.08)
        return True

    def do_move(self, width: float, speed: float = 0.10) -> bool:
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
            '/joint_states',
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

    def _js_callback(self, msg):
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

    def goto_joint_degree(self, joint_degrees):
        if not len(joint_degrees) == len(self.desired_seq):
            raise ValueError(f"目标关节数 {len(joint_degrees)} 与期望 {len(self.desired_seq)} 不匹配")
        current_gripper_state = self.get_state()[0:2]
        current_gripper_width = sum(current_gripper_state)
        joint_pos = joint_degrees[-7:]
        gripper_width = sum(joint_degrees[0:2])
        self.arm_client.goto_joint_degree(joint_pos)
        self.gripper_client.goto(gripper_width, current_gripper_width)

    def do_homing(self):
        self.arm_client.do_homing()
        self.gripper_client.do_homing()
        # while self.get_state() is None:
        #     self.executor.spin_once(timeout_sec=0.1)
        #     time.sleep(0.01)
        print("已收到第一条关节状态：", self.get_state())

    def goto_ee_pos(self, pos16):
        print("goto_ee_pos called with pos16:", pos16)
        self.arm_client.goto_ee_pos(pos16)

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
            elif message['command'] == 'goto_ee_pose':
                self.robot.goto_ee_pos(message['goal'])
                response = {'status': 'success', 'message': 'EE pose reached'}
            else:
                response = {'status': 'error', 'message': 'Unknown action'}
            self.socket.send_json(response)
