import os
import pathlib
import time

import hydra
import torch
import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf
from algorithms.utils.attr_dict import AttrDict

os.environ["WANDB_SILENT"] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

import numpy as np
import torch
import zmq
from termcolor import cprint

from roboverse_learn.algorithms.utils.multi_realsense import MultiRealsenseWrapper
from roboverse_learn.algorithms.utils.franka_ros_client import FrankaRobotClient

class RealWorldEnv:
    """
    The deployment is running on the local computer of the robot.
    """

    def __init__(
        self,
        device="gpu",
        img_size=224,
        num_points=4096,
        gripper_speed=1.0,
        gripper_force=0.1,
        use_server_robot=True
    ):
        # camera
        self.camera = MultiRealsenseWrapper()
        if use_server_robot:
            self.robot = FrankaRobotClient()
        else:
            self.robot = FrankaRobot()
        # inference device
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

    def step(self, single_step_action):
        # Stepping the robot
        action = self._action_to_ros_state(single_step_action)
        # import pdb; pdb.set_trace()
        self.robot.goto(action)
        # After execution
        cam_dict = self.camera()
        robot_state = torch.tensor(self.robot.get_state())
        obs_dict = {
            "agent_pos": robot_state.unsqueeze(0).to(self.device),
            "cameras": cam_dict
        }
        if not len(obs_dict["cameras"]["camera0"]["depth"].shape) == 3:
            obs_dict["cameras"]["camera0"]["depth"] = obs_dict["cameras"]["camera0"]["depth"].unsqueeze(-1)
        if (
            not obs_dict["cameras"]["camera0"]["rgb"].shape[-1] == 3
            or not obs_dict["cameras"]["camera0"]["depth"].shape[-1] == 1
            or not len(obs_dict["cameras"]["camera0"]["rgb"].shape) == 3
            or not len(obs_dict["cameras"]["camera0"]["depth"].shape) == 3
        ):
            raise ValueError(
                f"Please check the camera output shape. Expected RGB shape: (H, W, C) and Depth shape: (H, W, C), but got {obs_dict['cameras']['camera0']['rgb'].shape} and {obs_dict['cameras']['camera0']['depth'].shape}"
            )
        obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)
        obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0).to(self.device))
        obs_dict = AttrDict.from_dict(obs_dict)  # Convert the entire obs_dict to AttrDict for consistency
        return obs_dict

    def reset(self):
        # reset robot
        self.robot.do_homing()
        time.sleep(3)  # wait for the robot to finish homing
        print("Robot ready!")

        # ======== INIT ==========
        cam_dict = self.camera()
        robot_state = torch.tensor(self.robot.get_state())
        agent_pos = robot_state.to(self.device)
        obs_dict = {
            "agent_pos": robot_state.unsqueeze(0).to(self.device),
            "cameras": cam_dict
        }
        obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)
        obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0).to(self.device))
        obs_dict = AttrDict.from_dict(obs_dict)  # Convert the entire obs_dict to AttrDict for consistency
        return obs_dict

    def _robot_polymetis_state_to_tensor_state(self, robot_state):
        """
        Args:
            robot_state: polymetis_pb2.RobotState{
                timestamp: Dict
                joint_positions: tuple()
                joint_velocities: tuple()
                joint_torques_computed: tuple()
                prev_joint_torques_computed_safened: tuple()
                motor_torques_measured: tuple()
                motor_torques_external: tuple()
                motor_torques_desired: tuple()
                prev_controller_latency_ms: float
                prev_command_successful: bool
            }
        Returns:
            tensor_state: torch.Tensor([7,]): state_dim = 7
        """
        tensor_state = torch.tensor(robot_state.joint_positions[0:7]).to("cuda")  # (7,)
        return tensor_state

    def _gripper_polymetis_state_to_tensor_state(self, gripper_state):
        width = gripper_state.width
        return torch.tensor([width/2]*2).to("cuda")  # (2,)

    def _action_to_ros_state(self, action):
        """
        Args:
            action: Dict{robot_name: {'dof_pos_target':{"joint_name": tensor([1,])}}}
        Returns:
            robot_state: torch.Tensor([7,])
            gripper_width: float
        """
        if isinstance(action, list):
            if not len(action) == 1:
                raise ValueError(f"Expected action to be a list of length 1, but got {len(action)}")
            action = action[0]
        elif not isinstance(action, dict):
            raise ValueError(f"Expected action to be a dict or list, but got {type(action)}")

        robot_joint_name_sequence = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        gripper_joint_name_sequence = ["panda_finger_joint1", "panda_finger_joint2"]
        robot_name = "franka"
        robot_state = [
            action[robot_name]["dof_pos_target"][joint_name] for joint_name in robot_joint_name_sequence
        ]
        gripper_state = [
            action[robot_name]["dof_pos_target"][joint_name].item() for joint_name in gripper_joint_name_sequence
        ]
        action = gripper_state + robot_state
        return action
