import os
import pathlib
import time

import diffusion_policy.common.gr1_action_util as action_util
import diffusion_policy.common.rotation_util as rotation_util
import hydra
import torch
import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf

os.environ["WANDB_SILENT"] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

import numpy as np
import torch
from polymetis import GripperInterface, RobotInterface
from termcolor import cprint

from roboverse_learn.algorithms.utils.multi_realsense import MultiRealSense


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
    ):
        # camera
        self.camera = MultiRealSense(
            use_front_cam=True,  # by default we use single cam. but we also support multi-cam
            front_num_points=num_points,
            img_size=img_size,
        )
        self.robot = RobotInterface(ip_address="172.16.0.1")
        self.gripper = GripperInterface(ip_address="172.16.0.1")
        self.gripper_speed = gripper_speed
        self.gripper_force = gripper_force
        # inference device
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

    def step(self, single_step_action):
        # Stepping the robot
        robot_body_action, gripper_width = self._action_to_polymetis_state(single_step_action)
        self.robot.update_desired_joint_positions(robot_body_action)
        self.gripper.grasp(grasp_width=gripper_width, speed=self.gripper_speed, force=self.gripper_force)
        # After execution
        cam_dict = self.camera()
        agent_pos = self._robot_polymetis_state_to_tensor_state(self.robot.get_robot_state())
        obs_dict = {
            "agent_pos": torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
            "head_camera": cam_dict["head_camera"],
            "head_camera_depth": cam_dict["head_camera_depth"],
        }

        if (
            not obs_dict["head_camera"].shape[-1] == 3
            or not obs_dict["head_camera_depth"].shape[-1] == 1
            or not len(obs_dict["head_camera"].shape) == 3
            or not len(obs_dict["head_camera_depth"].shape) == 3
        ):
            raise ValueError(
                f"Please check the camera output shape. Expected RGB shape: (H, W, C) and Depth shape: (H, W, C), but got {obs_dict['head_camera'].shape} and {obs_dict['head_camera_depth'].shape}"
            )
        obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0).to(self.device))
        return obs_dict

    def reset(self):
        # reset robot
        self.robot.go_home()
        self.robot.start_joint_impedance()
        self.gripper.goto(width=0.08, speed=0.1, force=0.5)

        print("Robot ready!")

        # ======== INIT ==========
        cam_dict = self.camera()
        robot_state = self._robot_polymetis_state_to_tensor_state(self.robot.get_robot_state())
        obs_dict = {
            "head_camera": cam_dict["head_camera"],
            "head_camera_depth": cam_dict["head_camera_depth"],
            "agent_pos": robot_state,
        }
        obs_dict = dict_apply(obs_dict, lambda x: x.unsqueeze(0).to(self.device))
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
            tensor_state: torch.Tensor([9,]): state_dim = 9
        """
        tensor_state = torch.tensor(robot_state.joint_positions[0:9]).to("cuda")  # (9,)
        return tensor_state

    def _action_to_polymetis_state(self, action):
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
            action[robot_name]["dof_pos_target"][joint_name].item() for joint_name in robot_joint_name_sequence
        ]
        robot_state = torch.cat(robot_state, dim=0).to("cuda")  # (7,)
        gripper_width = [
            action[robot_name]["dof_pos_target"][joint_name].item() for joint_name in gripper_joint_name_sequence
        ]
        gripper_width = sum(gripper_width)  # assert two fingers are always symmetric
        return robot_state, gripper_width
